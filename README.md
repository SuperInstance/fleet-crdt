# fleet-crdt ⚒️

**Constraint-Native CRDT Merge Protocol for Distributed Fleet State**

A CRDT that preserves constraint satisfaction during merge — unlike Last-Write-Wins (LWW) or OR-Set, which discard structural knowledge about what makes a valid state.

## Why Not LWW?

Standard CRDTs are **value-blind**:

| Property | LWW-Register | OR-Set | **fleet-crdt** |
|---|---|---|---|
| Conflict resolution | Latest timestamp wins | Add/remove tracking | **Constraint maximalist** |
| Knowledge preservation | Zero — overwrites | Partial — tombstone | **Full — constraint topology** |
| Invariant guarantee | None | None | **Best-effort constraint satisfaction** |
| Multi-index constraints | Impossible | Impossible | **Native — pairwise bounds** |

When a fleet agent knows `|position[0] - position[1]| <= 5`, LWW will happily overwrite one index with a value that violates this. fleet-crdt evaluates candidates against all constraints and picks the one that preserves the maximum.

## How It Works

### Types

```rust
ConstraintState {
    agent_id: String,         // Which agent owns this
    values: Vec<i64>,          // The actual state vector
    constraints: Vec<Constraint>, // All known bounds
}

Constraint {
    index_a: usize,  // First index
    index_b: usize,  // Second index
    max_diff: i64,   // Bound: |values[a] - values[b]| <= max_diff
}

MergeResult {
    merged: ConstraintState,    // Resulting merged state
    conflicts_resolved: usize,  // Number of diverged indices
    constraints_preserved: bool, // All constraints still satisfied
}
```

### Merge Strategy

For each index where local and incoming differ:

1. **Score both candidates** — How many constraints does each value satisfy (using the original baseline to avoid chain-dependency)?
2. **Pick the maximalist** — The value that satisfies *more* constraints wins.
3. **Tie-break with average** — Equal scores → take `round((a + b) / 2)`.
4. **Defensive fallback** — Neither satisfies anything → keep local value.

This is a **constraint maximalist** strategy: prefer what keeps the most invariants intact, falling back to averaging or local-defensive when all options are equally bad.

### Example

```rust
let mut alice = CrdtNode::new("alice".into(), vec![10, 20, 30]);
alice.add_constraint(0, 1, 15).unwrap(); // |10-20| = 10 ≤ 15 ✓

let bob = CrdtNode::new("bob".into(), vec![10, 0, 30]);

let result = alice.merge(bob.state());
// Index 1: local=20 (score 1), incoming=0 (score 1) → tie → avg=10
assert_eq!(alice.state().values, vec![10, 10, 30]);
```

## API

| Method | Description |
|---|---|
| `CrdtNode::new(agent_id, values)` | Create a node |
| `node.add_constraint(a, b, max_diff)` | Add a pairwise bound |
| `node.local_set(index, value)` | Set a value (constraints checked after) |
| `node.merge(&state)` | Merge foreign state; returns `MergeResult` |
| `node.check_all_constraints()` | Verify all constraints hold |
| `node.state()` | Current `&ConstraintState` |
| `node.merge_history()` | All past merge logs |

## Properties

- **Zero external dependencies** — pure Rust, no serde, no tokio, no CRDT framework
- **Deterministic** — same inputs → same output (except clock increments)
- **Composable** — constraints from both sides are unioned during merge
- **Defensive** — never accepts a value that degrades constraint satisfaction when a better alternative exists
- **Independent resolution** — each index evaluated against the pre-merge baseline to avoid chain-dependency artifacts

## Limitations

1. **Per-index resolution is NP-hard for global CSPs** — a constraint spanning 3+ indices where each index conflicts cannot be optimally resolved index-by-index. For interdependent multi-index constraints, consider a CSP solver (not CRDT).
2. **Unbounded history** — `merge_history` grows with each merge. Prune or summarize for long-running systems.
3. **i64 values** — no floating-point, strings, or other types yet.

## License

MIT — use freely, ship boldly.
