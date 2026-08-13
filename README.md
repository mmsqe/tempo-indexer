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

Rows hold positions and filter keys, never transaction bodies — the node already stores
those. Disk grows with transaction *count* rather than size, and the index can be deleted
and rebuilt without touching the node's own state.

## Writes are atomic, because reorgs are not rare

`apply` takes a revert, a batch of rows and a tip, and commits them together. A crash
between a revert and its commit would otherwise leave orphaned rows under a tip claiming
they are current — a corruption no later write repairs.

The stored tip carries its block *hash*, not just a height. A node resuming after a
restart resolves where to continue from by hash, and after a reorg the block at a given
height is a different one.

## Usage

```rust
let store = tx_index::Store::open(datadir.join("indexer"))?;
let reader = store.reader();          // lock-free, cloneable, cannot write

store.apply(revert_from, &rows, tip)?;

let page = reader.query(&filter, after, tx_index::Order::Descending, 100)?;
```

`Store` is the writing half and does not clone, so "sole writer" is the type system's
problem; every read goes through `Reader`.

## Tests

`cargo test` — 21 tests over the keyspace layout, cursor round-tripping, filter
intersection, reorg handling and tip resumption. No network and no node: the store is
exercised directly.
