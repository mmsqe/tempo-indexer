//! A transaction index: RocksDB keyspaces mapping chain positions and filter keys.
//!
//! A node stores transactions by hash and by block, so it cannot cheaply answer the
//! question an explorer or wallet asks first — *which transactions involve this
//! address*. This is the secondary index that can.
//!
//! It deals in addresses, hashes and a type byte, and knows nothing about how a chain
//! represents a transaction. Keeping it that way is deliberate: maintaining the index
//! and serving it belong to a node, and this is only the store underneath — the part
//! that is the same everywhere, and the part where mistakes are expensive.
//!
//! The layout: a primary keyspace `block‖idx → row` and one secondary keyspace per
//! filterable column — `from`, `to`, `type` — each keyed `value‖block‖idx → ()`. Chain
//! position is the suffix everywhere because it is the only total order transactions
//! have: it keeps cursors stable (a page boundary cannot drift as new blocks arrive)
//! and makes every secondary range iterate in chain order for free. Rows hold positions
//! and filter keys, never transaction bodies — the node already stores those.

use std::path::Path;
use std::sync::Arc;

use alloy_primitives::{Address, TxHash, B256};
use rocksdb::{
    ColumnFamily, ColumnFamilyDescriptor, DBIteratorWithThreadMode, Direction, IteratorMode,
    Options, Snapshot, WriteBatch, DB,
};

/// Ordering direction for a query, mirroring the RPC's `sort.order`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Order {
    Ascending,
    #[default]
    Descending,
}

/// A transaction's position in the chain. Doubles as the pagination cursor.
///
/// `tx_index` is `u32` because that is what the key format stores; making the type
/// match removes a fallible conversion from every key builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Position {
    pub block_num: u64,
    pub tx_index: u32,
}

impl Position {
    pub const fn new(block_num: u64, tx_index: u32) -> Self {
        Self {
            block_num,
            tx_index,
        }
    }

    /// Zero-padded so cursors sort lexicographically too, keeping them opaque strings
    /// that clients never have to parse.
    pub fn encode(&self) -> String {
        format!("{:020}:{:010}", self.block_num, self.tx_index)
    }

    /// Parse a cursor from [`Self::encode`]; `None` on anything else, so a malformed
    /// cursor is an error rather than a silently different page.
    pub fn decode(cursor: &str) -> Option<Self> {
        let (block, index) = cursor.split_once(':')?;
        Some(Self {
            block_num: block.parse().ok()?,
            tx_index: index.parse().ok()?,
        })
    }
}

/// One indexed transaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedTx {
    pub position: Position,
    pub hash: TxHash,
    pub from: Address,
    /// `None` for contract creation.
    pub to: Option<Address>,
    pub tx_type: u8,
}

/// An inclusive window of block heights, either end open.
///
/// Heights rather than positions: a caller asking for "blocks 1000 to 2000" means whole
/// blocks, and the cursor is already the way to resume mid-block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockRange {
    pub min: Option<u64>,
    pub max: Option<u64>,
}

impl BlockRange {
    pub const fn new(min: Option<u64>, max: Option<u64>) -> Self {
        Self { min, max }
    }
}

/// Which transactions a query selects. Every field is optional and they intersect.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Filter {
    pub from: Option<Address>,
    pub to: Option<Address>,
    pub tx_type: Option<u8>,
    /// The heights to walk. Not a keyspace of its own: every key already ends in
    /// `block‖idx`, so a range is where each walk starts and stops.
    pub blocks: BlockRange,
}

/// The canonical block the index is caught up to — where a restart resumes from.
///
/// The hash is load-bearing: a node resolves where to resume by hash, and after a
/// reorg the block at a given height is a different one — a height alone would name
/// the wrong block exactly when resuming matters most.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tip {
    pub block_num: u64,
    pub hash: B256,
}

impl Tip {
    pub const fn new(block_num: u64, hash: B256) -> Self {
        Self { block_num, hash }
    }
}

/// Everything one change to the chain means for the index: what to drop, what to
/// insert, where the index stands afterwards.
///
/// Built by whoever watches the chain (a node's notification handler), consumed
/// whole by [`Store::apply`] — one struct, so the parts cannot be applied
/// separately and drift.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// Drop this block and everything above it.
    pub revert_from: Option<u64>,
    /// Rows to insert, in chain order.
    pub rows: Vec<IndexedTx>,
    /// Where the index stands afterwards — the resume point after a restart.
    pub tip: Option<Tip>,
    /// The committed tip, for acknowledging progress upstream. `None` for a pure
    /// revert: acknowledging one as progress lets the node prune history the index
    /// still needs replayed.
    pub committed: Option<Tip>,
}

impl Plan {
    /// Fold `next` into `self`, so that one [`Store::apply`] of the result leaves
    /// the index exactly as applying the two in order would.
    ///
    /// This is what lets an indexer that fell behind catch up in batches: at
    /// sub-second block times, notifications queue faster than one write apiece
    /// can drain them, so the watcher folds everything already queued into a
    /// single atomic write instead of replaying it block by block.
    /// `merge_matches_sequential_apply` below pins the equivalence.
    pub fn merge(&mut self, next: Plan) {
        if let Some(from) = next.revert_from {
            // Rows this plan would have inserted at or above the cut are orphaned
            // before ever reaching the store; the store's own rows are covered by
            // keeping the lowest cut.
            self.rows.retain(|row| row.position.block_num < from);
            self.revert_from = Some(self.revert_from.map_or(from, |f| f.min(from)));
            // A commit that only ever existed inside this merge was orphaned with
            // its rows, and must not be acknowledged as progress.
            if self.committed.is_some_and(|c| c.block_num >= from) {
                self.committed = None;
            }
        }
        self.rows.extend(next.rows);
        // Later knowledge wins; `or` keeps ours when `next` learned nothing new
        // (a pure revert carries a tip but no commit).
        self.tip = next.tip.or(self.tip);
        self.committed = next.committed.or(self.committed);
    }
}

const CF_PRIMARY: &str = "primary";
const CF_FROM: &str = "from";
const CF_TO: &str = "to";
const CF_TYPE: &str = "type";
const CFS: [&str; 4] = [CF_PRIMARY, CF_FROM, CF_TO, CF_TYPE];
/// The resume point: one key in the default keyspace, replaced on every applied tip.
const TIP_KEY: &[u8] = b"tip";

/// `block u64 BE ‖ idx u32 BE` — the primary key and every secondary's suffix.
fn pos_key(pos: Position) -> [u8; 12] {
    let mut k = [0u8; 12];
    k[..8].copy_from_slice(&pos.block_num.to_be_bytes());
    k[8..].copy_from_slice(&pos.tx_index.to_be_bytes());
    k
}

/// Decode a position from a key's last 12 bytes. An error, never a slice panic: a
/// wrong-length key only occurs through corruption or a hand-edited file — exactly
/// when the node should report rather than abort.
fn pos_from_key(suffix: &[u8]) -> eyre::Result<Position> {
    if suffix.len() != 12 {
        eyre::bail!("position key: expected 12 bytes, found {}", suffix.len());
    }
    Ok(Position::new(
        u64::from_be_bytes(suffix[..8].try_into().unwrap()),
        u32::from_be_bytes(suffix[8..].try_into().unwrap()),
    ))
}

fn sec_key(prefix: &[u8], pos: Position) -> Vec<u8> {
    let mut k = Vec::with_capacity(prefix.len() + 12);
    k.extend_from_slice(prefix);
    k.extend_from_slice(&pos_key(pos));
    k
}

/// `hash ‖ from ‖ type ‖ has_to ‖ [to]` — 54 or 74 bytes.
fn encode_row(row: &IndexedTx) -> Vec<u8> {
    let mut v = Vec::with_capacity(74);
    v.extend_from_slice(row.hash.as_slice());
    v.extend_from_slice(row.from.as_slice());
    v.push(row.tx_type);
    match &row.to {
        Some(to) => {
            v.push(1);
            v.extend_from_slice(to.as_slice());
        }
        None => v.push(0),
    }
    v
}

fn decode_row(position: Position, val: &[u8]) -> eyre::Result<IndexedTx> {
    let to = match (val.len(), val.get(53)) {
        (54, Some(0)) => None,
        (74, Some(1)) => Some(Address::from_slice(&val[54..74])),
        (n, flag) => eyre::bail!("row value: unexpected length {n} / to-flag {flag:?}"),
    };
    Ok(IndexedTx {
        position,
        hash: B256::from_slice(&val[..32]),
        from: Address::from_slice(&val[32..52]),
        to,
        tx_type: val[52],
    })
}

fn decode_tip(val: &[u8]) -> eyre::Result<Tip> {
    if val.len() != 40 {
        eyre::bail!("tip value: expected 40 bytes, found {}", val.len());
    }
    Ok(Tip::new(
        u64::from_be_bytes(val[..8].try_into().unwrap()),
        B256::from_slice(&val[8..]),
    ))
}

/// The writing half of the index. The ExEx owns it outright, and it does not clone,
/// so "sole writer" is the type system's problem rather than a convention. Reads for
/// the RPC go through [`Reader`].
#[derive(Debug)]
pub struct Store {
    db: Arc<DB>,
}

/// The read half, shared among concurrently running RPC handlers.
///
/// `Clone` and lock-free: RocksDB synchronizes readers internally, so no mutex
/// serializes queries and a query proceeds while the ExEx writes. It exposes no write
/// methods — a handler bug cannot write.
#[derive(Clone, Debug)]
pub struct Reader {
    db: Arc<DB>,
}

impl Store {
    /// Open (creating if absent) the index at `path`.
    pub fn open(path: impl AsRef<Path>) -> eyre::Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        let cfs = CFS.map(|name| {
            let mut o = Options::default();
            o.set_compression_type(rocksdb::DBCompressionType::Lz4);
            ColumnFamilyDescriptor::new(name, o)
        });
        let db = DB::open_cf_descriptors(&opts, path, cfs)?;
        Ok(Self { db: Arc::new(db) })
    }

    /// A shareable read handle onto the same database.
    pub fn reader(&self) -> Reader {
        Reader {
            db: self.db.clone(),
        }
    }

    /// Apply one [`Plan`] atomically: drop `revert_from` and above, insert `rows`,
    /// record the new tip.
    ///
    /// One `WriteBatch` on purpose. A reorg is a revert *and* a commit, and a crash
    /// between them would leave orphaned entries under a tip claiming they are
    /// current — a corruption no later notification repairs. Batch order also makes
    /// the overlap correct: a reorg's replacement rows are put after the range's
    /// deletes, so the puts win.
    ///
    /// A revert must scrub every keyspace by hand: a block's secondary entries are
    /// not contiguous (their prefix is an address or a type), so each is reconstructed
    /// from the primary row being dropped. Miss one and that keyspace serves ghosts —
    /// `revert_scrubs_every_secondary_keyspace` below pins all three.
    pub fn apply(&mut self, plan: &Plan) -> eyre::Result<()> {
        let (p, f, t, ty) = cfs(&self.db);
        let mut batch = WriteBatch::default();

        if let Some(from_block) = plan.revert_from {
            let start = pos_key(Position::new(from_block, 0));
            for item in self
                .db
                .iterator_cf(p, IteratorMode::From(&start, Direction::Forward))
            {
                let (key, val) = item?;
                let position = pos_from_key(&key)?;
                let row = decode_row(position, &val)?;
                batch.delete_cf(p, &key);
                batch.delete_cf(f, sec_key(row.from.as_slice(), position));
                if let Some(to) = &row.to {
                    batch.delete_cf(t, sec_key(to.as_slice(), position));
                }
                batch.delete_cf(ty, sec_key(&[row.tx_type], position));
            }
        }

        for row in &plan.rows {
            batch.put_cf(p, pos_key(row.position), encode_row(row));
            batch.put_cf(f, sec_key(row.from.as_slice(), row.position), []);
            if let Some(to) = &row.to {
                batch.put_cf(t, sec_key(to.as_slice(), row.position), []);
            }
            batch.put_cf(ty, sec_key(&[row.tx_type], row.position), []);
        }

        if let Some(tip) = plan.tip {
            let mut v = Vec::with_capacity(40);
            v.extend_from_slice(&tip.block_num.to_be_bytes());
            v.extend_from_slice(tip.hash.as_slice());
            batch.put(TIP_KEY, v);
        }

        // Default write options — WAL on, no fsync per batch. The index is derived
        // data: a crash that loses the last batches is repaired by backfilling from
        // the recorded tip.
        self.db.write(batch)?;
        Ok(())
    }

    /// The block the index is caught up to, or `None` before the first notification.
    ///
    /// Always canonical — a revert records the parent of what it dropped — so it can
    /// go straight back to the node as a resume head.
    pub fn indexed_tip(&self) -> eyre::Result<Option<Tip>> {
        read_tip(&self.db)
    }
}

impl Reader {
    /// Page through the index. `after` is exclusive and read in `order`'s direction,
    /// so paging forward never repeats or skips a row. Returns at most `limit` rows.
    ///
    /// The only query surface — [`Store`] deliberately has none, so everything that
    /// reads goes through a handle that cannot write.
    pub fn query(
        &self,
        filter: &Filter,
        after: Option<Position>,
        order: Order,
        limit: usize,
    ) -> eyre::Result<Vec<IndexedTx>> {
        // One snapshot for the whole page: a query is several reads — a walk per
        // filtered keyspace, then a lookup per row — and a revert committing
        // mid-query would delete a row a walk already named, turning its lookup
        // into an error. Pinning one commit makes the page the answer as of a moment.
        self.query_at(&self.db.snapshot(), filter, after, order, limit)
    }

    fn query_at(
        &self,
        snapshot: &Snapshot<'_>,
        filter: &Filter,
        after: Option<Position>,
        order: Order,
        limit: usize,
    ) -> eyre::Result<Vec<IndexedTx>> {
        // Every walk reads the same snapshot, cursor, range and direction; only
        // the keyspace and its prefix differ.
        let walk =
            |cf, prefix: &[u8]| scan(&self.db, snapshot, cf, prefix, after, filter.blocks, order);

        let mut scans = Vec::new();
        if let Some(from) = filter.from {
            scans.push(walk(CF_FROM, from.as_slice())?);
        }
        if let Some(to) = filter.to {
            scans.push(walk(CF_TO, to.as_slice())?);
        }
        if let Some(ty) = filter.tx_type {
            scans.push(walk(CF_TYPE, &[ty])?);
        }

        let primary = self.db.cf_handle(CF_PRIMARY).expect("cf created at open");

        // No filters: walk the primary directly — its values are the rows.
        if scans.is_empty() {
            let mut walk = walk(CF_PRIMARY, &[])?;
            let mut out = Vec::with_capacity(limit);
            while out.len() < limit {
                let Some((pos, val)) = walk.next_entry()? else {
                    break;
                };
                out.push(decode_row(pos, &val)?);
            }
            return Ok(out);
        }

        // Filtered: merge the secondary walks, then resolve rows from the primary.
        let positions = merge(scans, order, limit)?;
        let keys = positions.iter().map(|&p| (&primary, pos_key(p)));
        let mut out = Vec::with_capacity(positions.len());
        for (pos, got) in positions.iter().zip(snapshot.multi_get_cf(keys)) {
            let val = got?.ok_or_else(|| eyre::eyre!("secondary entry points at a missing row"))?;
            out.push(decode_row(*pos, &val)?);
        }
        Ok(out)
    }
}

/// Page size when the caller names none, and the ceiling on one that does.
///
/// Not arbitrary: `eth_getTransactions` publishes them — its `PaginationParams::limit`
/// documents "Defaults to 10. Maximum is 100". The ceiling is also what stops one
/// request asking a node to serialize the world.
pub const DEFAULT_LIMIT: usize = 10;
pub const MAX_LIMIT: usize = 100;

/// A page of the index; `next_cursor` is `None` on the last one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Page {
    pub rows: Vec<IndexedTx>,
    pub next_cursor: Option<String>,
}

impl Reader {
    /// One page of the index, with the cursor to continue from.
    ///
    /// The rules live here rather than in each caller because each is a way to
    /// truncate a walk without saying so:
    ///
    /// - `limit` clamps at both ends. A zero-row page has no last row to cut a
    ///   cursor from, so the caller hears the walk is over while rows remain.
    /// - One row *past* the limit decides whether another page exists. Counting
    ///   instead promises a next page that turns out empty whenever the total is an
    ///   exact multiple of the limit.
    /// - The cursor names the last row **returned**, never the one peeked at, or the
    ///   next page skips it.
    /// - A cursor this did not issue is an error, not an empty page, or a caller
    ///   replaying a stale one reads silence as the end.
    pub fn page(
        &self,
        filter: &Filter,
        cursor: Option<&str>,
        order: Order,
        limit: Option<usize>,
    ) -> eyre::Result<Page> {
        let after = match cursor {
            Some(cursor) => Some(
                Position::decode(cursor)
                    .ok_or_else(|| eyre::eyre!("malformed cursor: {cursor}"))?,
            ),
            None => None,
        };
        let limit = limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);

        let mut rows = self.query(filter, after, order, limit.saturating_add(1))?;
        let has_more = rows.len() > limit;
        rows.truncate(limit);
        let next_cursor = rows
            .last()
            .filter(|_| has_more)
            .map(|row| row.position.encode());

        Ok(Page { rows, next_cursor })
    }
}

type Cfs<'a> = (
    &'a ColumnFamily,
    &'a ColumnFamily,
    &'a ColumnFamily,
    &'a ColumnFamily,
);

fn cfs(db: &DB) -> Cfs<'_> {
    (
        db.cf_handle(CF_PRIMARY).expect("cf created at open"),
        db.cf_handle(CF_FROM).expect("cf created at open"),
        db.cf_handle(CF_TO).expect("cf created at open"),
        db.cf_handle(CF_TYPE).expect("cf created at open"),
    )
}

fn read_tip(db: &DB) -> eyre::Result<Option<Tip>> {
    db.get(TIP_KEY)?.map(|v| decode_tip(&v)).transpose()
}

/// How many entries a merge steps toward the other side before seeking straight to
/// it. Stepping is ~10× cheaper than a seek, so small gaps stay cheap; past this the
/// gap is large enough that a seek wins outright.
const GALLOP_AFTER: usize = 16;

/// A directional walk over one keyspace's `prefix‖block‖idx` range, yielding each
/// entry's position and value — the value matters on the primary keyspace, where it
/// is the row itself and saves a point lookup per hit.
struct KeyScan<'a> {
    snapshot: &'a Snapshot<'a>,
    cf: &'a ColumnFamily,
    it: DBIteratorWithThreadMode<'a, DB>,
    prefix: Vec<u8>,
    order: Order,
    /// The exact cursor key, skipped so the cursor is exclusive.
    skip: Option<Vec<u8>>,
    /// The last block this walk may yield, in `order`'s direction; `None` leaves
    /// that end open. The near end needs no field — it is where the walk seeked to.
    stop: Option<u64>,
}

impl KeyScan<'_> {
    fn next_entry(&mut self) -> eyre::Result<Option<(Position, Box<[u8]>)>> {
        for item in self.it.by_ref() {
            let (key, val) = item?;
            if !key.starts_with(&self.prefix) {
                break;
            }
            if self.skip.as_deref() == Some(&*key) {
                continue;
            }
            let pos = pos_from_key(&key[self.prefix.len()..])?;
            // Past the far end of the range; every later key is further past it,
            // so the walk is over.
            if self.past_stop(pos) {
                break;
            }
            return Ok(Some((pos, val)));
        }
        Ok(None)
    }

    fn past_stop(&self, pos: Position) -> bool {
        self.stop.is_some_and(|stop| match self.order {
            Order::Ascending => pos.block_num > stop,
            Order::Descending => pos.block_num < stop,
        })
    }

    fn next_pos(&mut self) -> eyre::Result<Option<Position>> {
        Ok(self.next_entry()?.map(|(pos, _)| pos))
    }

    /// True while `pos` has not yet reached `target` walking in `order`'s direction.
    fn still_ahead(&self, pos: Position, target: Position) -> bool {
        match self.order {
            Order::Descending => pos > target,
            Order::Ascending => pos < target,
        }
    }

    /// The first head at or past `target`: step up to [`GALLOP_AFTER`] entries, then
    /// give up walking and seek the iterator straight to the target key. This is what
    /// keeps an intersection O(smaller side) instead of O(larger side) — the measured
    /// difference between a 40-entry sender scanned against a 2000-entry recipient.
    fn advance_to(&mut self, target: Position) -> eyre::Result<Option<Position>> {
        for _ in 0..GALLOP_AFTER {
            match self.next_pos()? {
                None => return Ok(None),
                Some(pos) if self.still_ahead(pos, target) => continue,
                Some(pos) => return Ok(Some(pos)),
            }
        }
        let key = sec_key(&self.prefix, target);
        let direction = match self.order {
            Order::Ascending => Direction::Forward,
            Order::Descending => Direction::Reverse,
        };
        // Re-seek within the same snapshot, so a gallop cannot step from one
        // commit into the next.
        self.it = self
            .snapshot
            .iterator_cf(self.cf, IteratorMode::From(&key, direction));
        self.next_pos()
    }
}

fn scan<'a>(
    db: &'a DB,
    snapshot: &'a Snapshot<'a>,
    cf: &str,
    prefix: &[u8],
    after: Option<Position>,
    blocks: BlockRange,
    order: Order,
) -> eyre::Result<KeyScan<'a>> {
    let handle = db.cf_handle(cf).expect("cf created at open");
    // Where the range alone would start the walk — unbounded, the extreme position,
    // whose key is the same all-zero or all-`0xFF` seek an unbounded walk makes.
    let opening = match order {
        Order::Ascending => Position::new(blocks.min.unwrap_or(0), 0),
        Order::Descending => Position::new(blocks.max.unwrap_or(u64::MAX), u32::MAX),
    };
    // The cursor and the range both say where to begin, and the tighter one wins:
    // only the cursor, and a resumed page walks out of the range; only the range,
    // and rows the cursor already returned are replayed.
    let start = after.map_or(opening, |pos| match order {
        Order::Ascending => pos.max(opening),
        Order::Descending => pos.min(opening),
    });
    let direction = match order {
        Order::Ascending => Direction::Forward,
        Order::Descending => Direction::Reverse,
    };
    let stop = match order {
        Order::Ascending => blocks.max,
        Order::Descending => blocks.min,
    };
    Ok(KeyScan {
        snapshot,
        cf: handle,
        it: snapshot.iterator_cf(
            handle,
            IteratorMode::From(&sec_key(prefix, start), direction),
        ),
        prefix: prefix.to_vec(),
        order,
        skip: after.map(|pos| sec_key(prefix, pos)),
        stop,
    })
}

/// Intersect 1..=3 keyspace walks by sorted merge — every secondary shares the
/// `block‖idx` suffix ordering, which is what makes `AND` a merge at all.
fn merge(mut scans: Vec<KeyScan<'_>>, order: Order, limit: usize) -> eyre::Result<Vec<Position>> {
    let mut heads = Vec::with_capacity(scans.len());
    for s in &mut scans {
        heads.push(s.next_pos()?);
    }
    let mut out = Vec::new();
    while out.len() < limit && heads.iter().all(Option::is_some) {
        // The head *least* far along is the meeting point: anything ahead of it can
        // gallop straight down to it, and when every head sits on it, that position
        // is in the intersection.
        let iter = heads.iter().flatten();
        let target = match order {
            Order::Descending => *iter.min().unwrap(),
            Order::Ascending => *iter.max().unwrap(),
        };
        if heads.iter().all(|h| *h == Some(target)) {
            out.push(target);
            for (head, s) in heads.iter_mut().zip(&mut scans) {
                *head = s.next_pos()?;
            }
        } else {
            for (head, s) in heads.iter_mut().zip(&mut scans) {
                if *head != Some(target) {
                    *head = s.advance_to(target)?;
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn addr(byte: u8) -> Address {
        Address::from([byte; 20])
    }

    /// A deterministic tip for `block`, so a test can name one without inventing a hash.
    fn tip(block: u64) -> Option<Tip> {
        Some(Tip::new(block, B256::from([block as u8; 32])))
    }

    fn tx(block: u64, index: u32, from: u8, to: Option<u8>, tx_type: u8) -> IndexedTx {
        IndexedTx {
            position: Position::new(block, index),
            hash: B256::from([(block * 10 + u64::from(index)) as u8; 32]),
            from: addr(from),
            to: to.map(addr),
            tx_type,
        }
    }

    fn empty() -> (TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path().join("indexer")).unwrap();
        (dir, store)
    }

    /// A [`Plan`] for tests: insert `rows` (dropping from `revert_from` first) and
    /// move the tip.
    fn plan(revert_from: Option<u64>, rows: Vec<IndexedTx>, tip: Option<Tip>) -> Plan {
        Plan {
            revert_from,
            rows,
            tip,
            ..Default::default()
        }
    }

    fn seeded() -> (TempDir, Store) {
        let (dir, mut store) = empty();
        let entries = vec![
            tx(1, 0, 0xaa, Some(0xbb), 2),
            tx(1, 1, 0xaa, Some(0xcc), 2),
            tx(2, 0, 0xbb, Some(0xaa), 0),
            tx(3, 0, 0xaa, None, 2),
        ];
        store.apply(&plan(None, entries, tip(3))).unwrap();
        (dir, store)
    }

    #[test]
    fn cursor_round_trips() {
        let pos = Position::new(12345, 7);
        assert_eq!(Position::decode(&pos.encode()), Some(pos));
        assert_eq!(Position::decode("nonsense"), None, "no separator");
        assert_eq!(Position::decode("abc:def"), None, "non-numeric parts");
        assert_eq!(Position::decode("1:-1"), None, "negative index");
    }

    #[test]
    fn cursors_sort_lexicographically() {
        assert!(Position::new(2, 0).encode() > Position::new(1, 9).encode());
        assert!(Position::new(1, 2).encode() > Position::new(1, 1).encode());
    }

    #[test]
    fn zero_position_round_trips() {
        let pos = Position::new(0, 0);
        assert_eq!(Position::decode(&pos.encode()), Some(pos));
    }

    #[test]
    fn filters_by_sender() {
        let (_dir, store) = seeded();
        let filter = Filter {
            from: Some(addr(0xaa)),
            ..Default::default()
        };
        let got = store
            .reader()
            .query(&filter, None, Order::Ascending, 10)
            .unwrap();
        assert_eq!(got.len(), 3);
        assert!(got.iter().all(|t| t.from == addr(0xaa)));
    }

    #[test]
    fn filters_by_type() {
        let (_dir, store) = seeded();
        let filter = Filter {
            tx_type: Some(0),
            ..Default::default()
        };
        let got = store
            .reader()
            .query(&filter, None, Order::Ascending, 10)
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].position, Position::new(2, 0));
    }

    #[test]
    fn filters_intersect() {
        let (_dir, store) = seeded();
        let filter = Filter {
            from: Some(addr(0xaa)),
            to: Some(addr(0xcc)),
            ..Default::default()
        };
        let got = store
            .reader()
            .query(&filter, None, Order::Ascending, 10)
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].position, Position::new(1, 1));

        // Three-way, exercising the full merge width.
        let filter = Filter {
            from: Some(addr(0xaa)),
            to: Some(addr(0xbb)),
            tx_type: Some(2),
            ..Default::default()
        };
        let got = store
            .reader()
            .query(&filter, None, Order::Ascending, 10)
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].position, Position::new(1, 0));
    }

    /// The positions a whole query selects — what a range test is actually about.
    fn positions(store: &Store, filter: &Filter, order: Order) -> Vec<Position> {
        store
            .reader()
            .query(filter, None, order, MAX_LIMIT)
            .unwrap()
            .iter()
            .map(|t| t.position)
            .collect()
    }

    fn in_blocks(min: Option<u64>, max: Option<u64>) -> Filter {
        Filter {
            blocks: BlockRange::new(min, max),
            ..Default::default()
        }
    }

    #[test]
    fn a_block_range_bounds_the_walk() {
        let (_dir, store) = seeded();
        assert_eq!(
            positions(&store, &in_blocks(Some(2), Some(2)), Order::Ascending),
            vec![Position::new(2, 0)],
        );
        // Either end open.
        assert_eq!(
            positions(&store, &in_blocks(Some(2), None), Order::Ascending),
            vec![Position::new(2, 0), Position::new(3, 0)],
        );
        assert_eq!(
            positions(&store, &in_blocks(None, Some(1)), Order::Ascending),
            vec![Position::new(1, 0), Position::new(1, 1)],
        );
        // Both ends open is exactly the unbounded walk.
        assert_eq!(
            positions(&store, &in_blocks(None, None), Order::Ascending),
            positions(&store, &Filter::default(), Order::Ascending),
        );
    }

    #[test]
    fn a_block_range_reads_the_same_both_ways() {
        let (_dir, store) = seeded();
        let filter = in_blocks(Some(1), Some(2));
        let mut descending = positions(&store, &filter, Order::Descending);
        descending.reverse();
        assert_eq!(descending.len(), 3);
        assert_eq!(positions(&store, &filter, Order::Ascending), descending);
    }

    #[test]
    fn a_block_range_bounds_a_filtered_walk() {
        // The bound has to hold on the merged path too, where each secondary walk
        // stops on its own rather than one primary scan stopping for everyone.
        let (_dir, store) = seeded();
        let filter = Filter {
            from: Some(addr(0xaa)),
            blocks: BlockRange::new(None, Some(1)),
            ..Default::default()
        };
        assert_eq!(
            positions(&store, &filter, Order::Ascending),
            vec![Position::new(1, 0), Position::new(1, 1)],
            "0xaa also sends in block 3, outside the range",
        );
    }

    #[test]
    fn an_impossible_range_selects_nothing() {
        let (_dir, store) = seeded();
        // Past the tip, and a max below its own min.
        for filter in [in_blocks(Some(10), Some(20)), in_blocks(Some(3), Some(1))] {
            for order in [Order::Ascending, Order::Descending] {
                assert!(positions(&store, &filter, order).is_empty(), "{filter:?}");
            }
        }
    }

    #[test]
    fn a_range_and_a_cursor_take_the_tighter_bound() {
        // Both say where to begin. Honouring only the cursor walks out of the range;
        // honouring only the range replays rows the cursor already returned.
        let (_dir, store) = seeded();
        let reader = store.reader();
        let at = |filter: Filter, after, order| {
            reader
                .query(&filter, Some(after), order, MAX_LIMIT)
                .unwrap()
                .iter()
                .map(|t| t.position)
                .collect::<Vec<_>>()
        };

        assert_eq!(
            at(
                in_blocks(Some(3), None),
                Position::new(1, 0),
                Order::Ascending
            ),
            vec![Position::new(3, 0)],
            "the range is the tighter opening",
        );
        assert_eq!(
            at(
                in_blocks(Some(1), Some(2)),
                Position::new(1, 0),
                Order::Ascending
            ),
            vec![Position::new(1, 1), Position::new(2, 0)],
            "the cursor is the tighter opening, and the range still ends the walk",
        );
        assert_eq!(
            at(
                in_blocks(None, Some(1)),
                Position::new(3, 0),
                Order::Descending
            ),
            vec![Position::new(1, 1), Position::new(1, 0)],
            "descending, the range's max is the tighter opening",
        );
    }

    #[test]
    fn paging_within_a_range_stays_inside_it() {
        let (_dir, store) = seeded();
        let reader = store.reader();
        let filter = in_blocks(Some(1), Some(2));
        let all = reader
            .page(&filter, None, Order::Ascending, Some(MAX_LIMIT))
            .unwrap();
        let (mut seen, mut cursor) = (Vec::new(), None);
        for _ in 0..=all.rows.len() {
            let page = reader
                .page(&filter, cursor.as_deref(), Order::Ascending, Some(1))
                .unwrap();
            seen.extend(page.rows);
            cursor = page.next_cursor;
            if cursor.is_none() {
                break;
            }
        }
        assert!(cursor.is_none(), "the cursor never ran out");
        assert_eq!(seen, all.rows);
        assert!(seen.iter().all(|row| row.position.block_num <= 2));
    }

    #[test]
    fn a_gallop_stops_at_the_range_rather_than_seeking_past_it() {
        // The one place a range and the intersection meet: `advance_to` gives up
        // stepping and seeks straight to its target, and that target can sit outside
        // the range. Checking the bound only where the walk *starts* would let the
        // seek land beyond it and serve a row the caller excluded.
        let (_dir, mut store) = empty();
        // Filler to the shared recipient, more than GALLOP_AFTER of it, so the
        // recipient's walk gives up stepping and seeks.
        let mut rows: Vec<_> = (1..=39).map(|b| tx(b, 0, 0xbb, Some(0xcc), 2)).collect();
        // The rare sender inside the range, but paying someone else: the seek target
        // is in range while the entry the recipient's walk lands on is not.
        rows.push(tx(40, 0, 0x0a, Some(0xdd), 2));
        // The pair's only real meeting, past the range's end.
        rows.push(tx(60, 0, 0x0a, Some(0xcc), 2));
        store.apply(&plan(None, rows, tip(60))).unwrap();

        let filter = |blocks| Filter {
            from: Some(addr(0x0a)),
            to: Some(addr(0xcc)),
            blocks,
            ..Default::default()
        };
        assert_eq!(
            positions(&store, &filter(BlockRange::default()), Order::Ascending),
            vec![Position::new(60, 0)],
            "unbounded, the gallop finds it",
        );
        assert!(
            positions(
                &store,
                &filter(BlockRange::new(None, Some(50))),
                Order::Ascending
            )
            .is_empty(),
            "the seek landed past the range's end and the walk must stop there",
        );
    }

    #[test]
    fn contract_creation_has_no_recipient() {
        let (_dir, store) = seeded();
        let got = store
            .reader()
            .query(&Filter::default(), None, Order::Ascending, 10)
            .unwrap();
        let creation = got.iter().find(|t| t.position.block_num == 3).unwrap();
        assert_eq!(creation.to, None);
    }

    #[test]
    fn descending_is_the_reverse_of_ascending() {
        let (_dir, store) = seeded();
        let asc = store
            .reader()
            .query(&Filter::default(), None, Order::Ascending, 10)
            .unwrap();
        let mut desc = store
            .reader()
            .query(&Filter::default(), None, Order::Descending, 10)
            .unwrap();
        desc.reverse();
        assert_eq!(asc, desc);
    }

    #[test]
    fn paging_visits_every_row_exactly_once() {
        let (_dir, store) = seeded();
        for order in [Order::Ascending, Order::Descending] {
            let all = store
                .reader()
                .query(&Filter::default(), None, order, 100)
                .unwrap();
            let mut seen = Vec::new();
            let mut cursor = None;
            loop {
                let page = store
                    .reader()
                    .query(&Filter::default(), cursor, order, 1)
                    .unwrap();
                let Some(entry) = page.into_iter().next() else {
                    break;
                };
                cursor = Some(entry.position);
                seen.push(entry);
            }
            assert_eq!(
                seen, all,
                "paged result diverged from the single-page result"
            );
        }
    }

    #[test]
    fn filtered_paging_pages_through_the_merge() {
        // The cursor has to hold on the merged path too, not only the primary scan.
        let (_dir, store) = seeded();
        let filter = Filter {
            from: Some(addr(0xaa)),
            ..Default::default()
        };
        let all = store
            .reader()
            .query(&filter, None, Order::Descending, 100)
            .unwrap();
        let mut seen = Vec::new();
        let mut cursor = None;
        loop {
            let page = store
                .reader()
                .query(&filter, cursor, Order::Descending, 1)
                .unwrap();
            let Some(entry) = page.into_iter().next() else {
                break;
            };
            cursor = Some(entry.position);
            seen.push(entry);
        }
        assert_eq!(seen, all);
    }

    #[test]
    fn revert_drops_the_block_and_everything_above() {
        let (_dir, mut store) = seeded();
        store.apply(&plan(Some(2), vec![], tip(1))).unwrap();
        let left = store
            .reader()
            .query(&Filter::default(), None, Order::Ascending, 10)
            .unwrap();
        assert_eq!(left.len(), 2);
        assert!(left.iter().all(|t| t.position.block_num == 1));
        assert_eq!(store.indexed_tip().unwrap(), tip(1));
    }

    #[test]
    fn revert_scrubs_every_secondary_keyspace() {
        // The KV-specific hazard a SQL DELETE never had: each secondary entry is
        // reconstructed from its primary row. A missed one serves ghosts.
        let (_dir, mut store) = seeded();
        store.apply(&plan(Some(1), vec![], tip(0))).unwrap();
        for filter in [
            Filter {
                from: Some(addr(0xaa)),
                ..Default::default()
            },
            Filter {
                to: Some(addr(0xbb)),
                ..Default::default()
            },
            Filter {
                tx_type: Some(2),
                ..Default::default()
            },
        ] {
            assert!(
                store
                    .reader()
                    .query(&filter, None, Order::Ascending, 10)
                    .unwrap()
                    .is_empty(),
                "revert left a ghost behind {filter:?}"
            );
        }
    }

    #[test]
    fn intersection_gallops_across_a_wide_gap() {
        // Forces the seek path: the recipient's keyspace holds far more than
        // GALLOP_AFTER entries between the sender's, so stepping alone never gets
        // there. Results must match the brute-force answer in both directions.
        let (_dir, mut store) = empty();
        let mut rows = Vec::new();
        // Blocks 1..=2: the rare sender pays the shared recipient.
        for b in 1..=2u64 {
            rows.push(tx(b, 0, 0x0a, Some(0xcc), 2));
        }
        // Blocks 3..=60: filler traffic to the same recipient from someone else.
        for b in 3..=60u64 {
            rows.push(tx(b, 0, 0xbb, Some(0xcc), 2));
        }
        store.apply(&plan(None, rows, tip(60))).unwrap();

        let filter = Filter {
            from: Some(addr(0x0a)),
            to: Some(addr(0xcc)),
            ..Default::default()
        };
        let want: Vec<Position> = vec![Position::new(1, 0), Position::new(2, 0)];
        for order in [Order::Ascending, Order::Descending] {
            let got: Vec<Position> = store
                .reader()
                .query(&filter, None, order, 100)
                .unwrap()
                .iter()
                .map(|t| t.position)
                .collect();
            let mut expect = want.clone();
            if order == Order::Descending {
                expect.reverse();
            }
            assert_eq!(got, expect, "gallop diverged from brute force ({order:?})");
        }
    }

    #[test]
    fn reorg_replaces_rather_than_merges() {
        let (_dir, mut store) = seeded();
        let replacement = vec![tx(2, 0, 0xdd, Some(0xee), 2)];
        store.apply(&plan(Some(2), replacement, tip(3))).unwrap();

        let orphaned = store
            .reader()
            .query(
                &Filter {
                    from: Some(addr(0xbb)),
                    ..Default::default()
                },
                None,
                Order::Ascending,
                10,
            )
            .unwrap();
        assert!(orphaned.is_empty(), "orphaned transaction survived a reorg");

        let canonical = store
            .reader()
            .query(
                &Filter {
                    from: Some(addr(0xdd)),
                    ..Default::default()
                },
                None,
                Order::Ascending,
                10,
            )
            .unwrap();
        assert_eq!(canonical.len(), 1);
    }

    #[test]
    fn reapplying_a_block_does_not_duplicate_it() {
        let (_dir, mut store) = seeded();
        store
            .apply(&plan(None, vec![tx(1, 0, 0xaa, Some(0xbb), 2)], tip(3)))
            .unwrap();
        let got = store
            .reader()
            .query(&Filter::default(), None, Order::Ascending, 10)
            .unwrap();
        assert_eq!(got.len(), 4);
    }

    #[test]
    fn tip_is_absent_before_the_first_notification() {
        let (_dir, store) = empty();
        assert_eq!(store.indexed_tip().unwrap(), None);
    }

    #[test]
    fn tip_round_trips_with_its_hash() {
        let (_dir, mut store) = empty();
        let want = Tip::new(7, B256::from([0x7c; 32]));
        store.apply(&plan(None, vec![], Some(want))).unwrap();
        assert_eq!(store.indexed_tip().unwrap(), Some(want));
    }

    #[test]
    fn tip_is_replaced_not_appended() {
        let (_dir, mut store) = empty();
        store.apply(&plan(None, vec![], tip(1))).unwrap();
        store.apply(&plan(None, vec![], tip(2))).unwrap();
        assert_eq!(store.indexed_tip().unwrap(), tip(2));
    }

    #[test]
    fn a_revert_moves_the_tip_back() {
        let (_dir, mut store) = seeded();
        assert_eq!(store.indexed_tip().unwrap(), tip(3));
        store.apply(&plan(Some(2), vec![], tip(1))).unwrap();
        assert_eq!(store.indexed_tip().unwrap(), tip(1));
    }

    #[test]
    fn reader_sees_the_writers_commits() {
        // "Cannot write" needs no runtime test: `Reader` has no write methods, so it
        // is a compile error instead.
        let (_dir, mut store) = empty();
        let reader = store.reader();
        store
            .apply(&plan(None, vec![tx(1, 0, 0xaa, Some(0xbb), 2)], tip(1)))
            .unwrap();
        let got = reader
            .query(&Filter::default(), None, Order::Ascending, 10)
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(read_tip(&reader.db).unwrap(), tip(1));
    }

    #[test]
    fn a_page_reads_one_commit_even_as_the_writer_reverts() {
        // Without a pinned snapshot, a revert landing mid-query deletes a row the
        // walk already named, and its lookup errors — on the path every reorg takes.
        let (_dir, mut store) = seeded();
        let reader = store.reader();
        let snapshot = reader.db.snapshot();

        // Everything the query is about to name is dropped *after* it is pinned.
        store.apply(&plan(Some(1), vec![], tip(0))).unwrap();

        let filter = Filter {
            from: Some(addr(0xaa)),
            ..Default::default()
        };
        let pinned = reader
            .query_at(&snapshot, &filter, None, Order::Ascending, 10)
            .unwrap();
        assert_eq!(
            pinned.len(),
            3,
            "the walk and the lookups that resolve it must read the same commit"
        );

        // A query started afterwards pins the later commit, and sees the revert.
        assert!(reader
            .query(&filter, None, Order::Ascending, 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_corrupt_row_errors_rather_than_panicking() {
        // Only reachable through corruption or a hand-edited file, but the blow-up
        // would otherwise be a slice panic inside the node.
        let (_dir, store) = seeded();
        let p = store.db.cf_handle(CF_PRIMARY).unwrap();
        store
            .db
            .put_cf(p, pos_key(Position::new(1, 0)), b"short")
            .unwrap();
        store.db.put(TIP_KEY, b"also short").unwrap();

        assert!(store
            .reader()
            .query(&Filter::default(), None, Order::Ascending, 10)
            .is_err());
        assert!(store.indexed_tip().is_err());
    }
    #[test]
    fn a_page_stops_at_the_limit_and_says_where_to_resume() {
        let (_dir, store) = seeded();
        let page = store
            .reader()
            .page(&Filter::default(), None, Order::Ascending, Some(2))
            .unwrap();
        assert_eq!(page.rows.len(), 2);
        // The cursor names the last row returned, not the row peeked at to learn
        // there was more -- resuming from the peeked one would skip it.
        assert_eq!(page.next_cursor, Some(page.rows[1].position.encode()));
    }

    #[test]
    fn the_last_page_reports_no_cursor() {
        // Exactly as many rows as the limit, so the look-ahead comes back empty. A
        // page sized from a count would hand back a cursor onto nothing here, and
        // every caller would pay one empty round trip to discover the end.
        let (_dir, store) = seeded();
        let page = store
            .reader()
            .page(&Filter::default(), None, Order::Ascending, Some(4))
            .unwrap();
        assert_eq!(page.rows.len(), 4);
        assert_eq!(page.next_cursor, None);
    }

    #[test]
    fn paging_by_cursor_covers_every_row_once() {
        let (_dir, store) = seeded();
        let reader = store.reader();
        for order in [Order::Ascending, Order::Descending] {
            let all = reader
                .page(&Filter::default(), None, order, Some(100))
                .unwrap();
            let (mut seen, mut cursor) = (Vec::new(), None);
            // Bounded: a cursor that fails to advance would otherwise hang rather
            // than fail, which is the harder of the two to read.
            for _ in 0..=all.rows.len() {
                let page = reader
                    .page(&Filter::default(), cursor.as_deref(), order, Some(1))
                    .unwrap();
                seen.extend(page.rows);
                cursor = page.next_cursor;
                if cursor.is_none() {
                    break;
                }
            }
            assert!(cursor.is_none(), "the cursor never ran out ({order:?})");
            assert_eq!(seen, all.rows, "paged result diverged ({order:?})");
        }
    }

    #[test]
    fn limit_clamps_at_both_ends() {
        let (_dir, store) = seeded();
        let reader = store.reader();
        let page = |limit| {
            reader
                .page(&Filter::default(), None, Order::Ascending, limit)
                .unwrap()
        };
        // Zero would leave no last row to cut a cursor from, so it clamps up rather
        // than reporting a walk that is over.
        assert_eq!(page(Some(0)).rows.len(), 1);
        assert!(page(Some(0)).next_cursor.is_some());
        assert_eq!(page(Some(usize::MAX)).rows.len(), 4, "clamped to MAX_LIMIT");
        assert_eq!(page(None).rows.len(), 4, "fewer rows than DEFAULT_LIMIT");
    }

    #[test]
    fn a_cursor_this_did_not_issue_is_rejected() {
        // Not an empty page: a caller replaying a stale cursor would read silence as
        // the end of the walk.
        let (_dir, store) = seeded();
        let got = store.reader().page(
            &Filter::default(),
            Some("cursor-from-somewhere-else"),
            Order::Ascending,
            None,
        );
        assert!(got.is_err());
    }

    /// A commit plan for `rows` ending at `tip_block`, as a watcher would build it.
    fn commit(rows: Vec<IndexedTx>, tip_block: u64) -> Plan {
        Plan {
            revert_from: None,
            rows,
            tip: tip(tip_block),
            committed: tip(tip_block),
        }
    }

    #[test]
    fn merge_concatenates_commits() {
        let mut merged = commit(vec![tx(1, 0, 0xaa, None, 2)], 1);
        merged.merge(commit(vec![tx(2, 0, 0xbb, None, 2)], 2));
        assert_eq!(
            merged,
            Plan {
                revert_from: None,
                rows: vec![tx(1, 0, 0xaa, None, 2), tx(2, 0, 0xbb, None, 2)],
                tip: tip(2),
                committed: tip(2),
            }
        );
    }

    #[test]
    fn merge_reorg_cuts_unwritten_rows() {
        // Blocks 2 and 3 exist only inside the merge; the reorg at 2 orphans them
        // before they ever reach the store, and the replacement row wins.
        let mut merged = commit(vec![tx(1, 0, 0xaa, None, 2), tx(2, 0, 0xbb, None, 2)], 2);
        merged.merge(commit(vec![tx(3, 0, 0xcc, None, 2)], 3));
        merged.merge(Plan {
            revert_from: Some(2),
            rows: vec![tx(2, 0, 0xdd, None, 0)],
            tip: tip(2),
            committed: tip(2),
        });
        assert_eq!(
            merged,
            Plan {
                revert_from: Some(2),
                rows: vec![tx(1, 0, 0xaa, None, 2), tx(2, 0, 0xdd, None, 0)],
                tip: tip(2),
                committed: tip(2),
            }
        );
    }

    #[test]
    fn merge_keeps_lowest_cut() {
        // Two reverts: the store must be scrubbed from the lower of the two, no
        // matter the order they arrive in.
        let mut merged = plan(Some(5), vec![], tip(4));
        merged.merge(plan(Some(3), vec![], tip(2)));
        assert_eq!(merged.revert_from, Some(3));

        let mut merged = plan(Some(3), vec![], tip(2));
        merged.merge(plan(Some(5), vec![], tip(4)));
        assert_eq!(merged.revert_from, Some(3));
    }

    #[test]
    fn merge_revert_only_drops_batch_commit() {
        // The commit at 2 was orphaned inside the merge; acknowledging it upstream
        // would let the node prune history the index still needs replayed.
        let mut merged = commit(vec![tx(2, 0, 0xaa, None, 2)], 2);
        merged.merge(plan(Some(2), vec![], tip(1)));
        assert_eq!(merged.committed, None);
        assert_eq!(merged.tip, tip(1), "tip still moves to the revert's parent");
        assert!(merged.rows.is_empty());
    }

    #[test]
    fn merge_keeps_a_commit_below_the_cut() {
        // The batch committed block 2, then a queued reorg cut from block 5. Block 2
        // is still canonical, and the merged apply is this batch's only chance to
        // acknowledge it -- dropping it is safe but pays a replay for nothing.
        let mut merged = Plan {
            rows: vec![tx(2, 0, 0xaa, None, 2)],
            tip: tip(2),
            committed: tip(2),
            ..Default::default()
        };
        merged.merge(Plan {
            revert_from: Some(5),
            tip: tip(4),
            ..Default::default()
        });
        assert_eq!(
            merged.committed,
            tip(2),
            "a commit below the cut survived the merge"
        );
        assert_eq!(merged.tip, tip(4));
    }

    #[test]
    fn merge_matches_sequential_apply() {
        // The contract `merge` exists for: one apply of the fold leaves the store
        // exactly as applying each plan in order would — including a reorg that
        // cuts into rows the store already holds.
        let history = || {
            vec![
                commit(
                    vec![tx(1, 0, 0xaa, Some(0xbb), 2), tx(1, 1, 0xaa, None, 0)],
                    1,
                ),
                commit(vec![tx(2, 0, 0xbb, Some(0xaa), 2)], 2),
                commit(vec![tx(3, 0, 0xcc, None, 2)], 3),
                // Reorg: 2 and 3 are replaced by a different block 2.
                Plan {
                    revert_from: Some(2),
                    rows: vec![tx(2, 0, 0xdd, Some(0xee), 0)],
                    tip: Some(Tip::new(2, B256::from([0x22; 32]))),
                    committed: Some(Tip::new(2, B256::from([0x22; 32]))),
                },
                commit(vec![tx(3, 0, 0xee, Some(0xdd), 2)], 3),
            ]
        };

        let (_seq_dir, mut sequential) = seeded();
        for plan in history() {
            sequential.apply(&plan).unwrap();
        }

        let (_merged_dir, mut merged_store) = seeded();
        let mut folded = Plan::default();
        for plan in history() {
            folded.merge(plan);
        }
        merged_store.apply(&folded).unwrap();

        let all = |store: &Store| {
            store
                .reader()
                .query(&Filter::default(), None, Order::Ascending, MAX_LIMIT)
                .unwrap()
        };
        assert_eq!(all(&sequential), all(&merged_store));
        assert_eq!(
            sequential.indexed_tip().unwrap(),
            merged_store.indexed_tip().unwrap()
        );
        // The secondaries were scrubbed identically too — query each one.
        for filter in [
            Filter {
                from: Some(addr(0xbb)),
                ..Default::default()
            },
            Filter {
                to: Some(addr(0xaa)),
                ..Default::default()
            },
            Filter {
                tx_type: Some(0),
                ..Default::default()
            },
        ] {
            let seq = sequential
                .reader()
                .query(&filter, None, Order::Ascending, MAX_LIMIT);
            let mrg = merged_store
                .reader()
                .query(&filter, None, Order::Ascending, MAX_LIMIT);
            assert_eq!(seq.unwrap(), mrg.unwrap());
        }
    }
}
