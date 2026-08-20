# tx-index

A secondary index from address and transaction type to chain position, for nodes that
store transactions by hash and by block.

A node can answer "give me transaction `0x…`" and "give me block 12". It cannot cheaply
answer the question an explorer or wallet asks first — *which transactions involve this
address* — because nothing is keyed that way. This is the index that answers it.

It deals in `Address`, `TxHash` and a type byte, and knows nothing about how a chain
represents a transaction: maintaining and serving the index belong to the node, and this
is only the store underneath — the part that is the same everywhere, and the part where
mistakes are expensive.

## Layout

RocksDB, one primary keyspace and one secondary per filterable column:

```
primary   block‖idx           → hash, from, type, fee_token?, fee_payer?, to*
from      from ‖block‖idx     → ()
to        to   ‖block‖idx     → ()
type      type ‖block‖idx     → ()
fee_token fee_token‖block‖idx → ()
fee_payer fee_payer‖block‖idx → ()
count     kind‖0‖value        → u64
```

Chain position is the suffix everywhere, because it is the only *total* order
transactions have — block number alone repeats within a block. That one choice buys
three things: cursors are stable, so a page boundary cannot drift as new blocks arrive;
every secondary range iterates in chain order for free; and filters intersect by merging
sorted ranges rather than by loading and sorting.

Rows hold positions and filter keys, never transaction bodies — the node already stores
those. Disk grows with transaction *count* rather than size, and the index can be deleted
and rebuilt without touching the node's own state.

A transaction has one `to` entry per *distinct address it calls*. A chain may batch, and
an index that recorded only the first target answers a `to` filter on any of the others
with silence — which reads exactly like "no such transactions".

`fee_token` and `fee_payer` are for chains where gas can be paid in a chosen token or by
someone other than the sender; where it cannot, both are unset and neither keyspace is
written. `fee_payer` records who paid for *every* transaction, the sender included: a
filter answering "who paid" for some and nothing for the rest leaves no way to ask for
the half it dropped. That costs one entry per transaction.

Two things are deliberately *not* keyspaces. A block range is where each walk seeks to
and where it stops, since every key already ends in `block‖idx`. And "did this address
take part" is the `from` and `to` walks unioned — the same sorted merge as the
intersection around it; a keyspace of its own would answer in one walk instead of two, at
an extra entry per transaction on every write, and the read is the cheaper side to pay on.

`count` is the one exception to "a keyspace per filterable column": RocksDB has no cheap
`COUNT`, so "how many does this address have" is maintained as rows are written. It is
kept per filter value, so an intersection, or a block range, has no stored answer and
`Reader::count` says `None` rather than inventing one.

## Writes are atomic, because reorgs are not rare

`apply` takes one `Plan` — a revert, a batch of rows and a tip — and commits it whole. A
crash between a revert and its commit would otherwise leave orphaned rows under a tip
claiming they are current — a corruption no later write repairs.

A watcher that has fallen behind folds every queued plan into one (`Plan::merge`) and
applies that: notifications can arrive faster than a write apiece can drain them, and
one merged apply leaves the index exactly where replaying each plan in order would.

The stored tip carries its block *hash*, not just a height. A node resuming after a
restart resolves where to continue from by hash, and after a reorg the block at a given
height is a different one.

## Usage

```rust
let mut store = tx_index::Store::open(datadir.join("indexer"))?;
let reader = store.reader();          // lock-free, cloneable, cannot write

store.apply(&plan)?;                  // one notification's revert, rows and tip

let page = reader.page(&filter, cursor, tx_index::Order::Descending, Some(100))?;
```

`Store` is the writing half and does not clone, so "sole writer" is the type system's
problem; every read goes through `Reader`.

`Reader::page` wraps `query` with the paging rules — the limit clamped at both ends, a
look-ahead row to decide whether another page exists, and a cursor cut from the last row
*returned*. Each is a way to silently truncate a walk, so they live here rather than in
each caller.

## What the index is holding

```rust
let stats = store.reader().stats()?;  // tip, row count, bytes per keyspace
```

On `Reader` because it only reads. There is no `compact`: RocksDB's own compaction
already reclaims what this store leaves behind, and the two chains using it finalize, so
a revert reaches only as deep as the unfinalized head — a handful of scattered tombstones,
not a rewrite worth blocking a node for. `Stats::bytes` reports *file* bytes, which run
ahead of what is reachable until that automatic compaction catches up.

## Tests

`cargo test` — the keyspace layout, cursor round-tripping, filter intersection, paging,
reorg handling, plan folding and tip resumption. No network and no node: the store is
exercised directly.
