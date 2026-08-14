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
primary   block‖idx           → hash, from, type, to?
from      from ‖block‖idx     → ()
to        to   ‖block‖idx     → ()
type      type ‖block‖idx     → ()
```

Chain position is the suffix everywhere, because it is the only *total* order
transactions have — block number alone repeats within a block. That one choice buys
three things: cursors are stable, so a page boundary cannot drift as new blocks arrive;
every secondary range iterates in chain order for free; and filters intersect by merging
sorted ranges rather than by loading and sorting.

A block range is not a fourth keyspace for the same reason: every key already ends in
`block‖idx`, so `fromBlock`/`toBlock` is where each walk seeks to and where it stops.

Neither is "did this address take part", the first query a wallet makes: it is the
`from` and `to` walks unioned — the same sorted merge as the intersection around it. A
keyspace of its own would answer it in one walk instead of two, but cost an extra entry
per transaction on every write; the read is the cheaper side to pay on.

Rows hold positions and filter keys, never transaction bodies — the node already stores
those. Disk grows with transaction *count* rather than size, and the index can be deleted
and rebuilt without touching the node's own state.

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

## Tests

`cargo test` — the keyspace layout, cursor round-tripping, filter intersection, paging,
reorg handling, plan folding and tip resumption. No network and no node: the store is
exercised directly.
