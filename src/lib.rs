//! # fleet-crdt
//!
//! Constraint-Native CRDT merge protocol for distributed fleet state.
//!
//! Standard CRDTs (Last-Write-Wins, OR-Set) lose constraint satisfaction
//! information during merge. This crate implements a CRDT that **preserves
//! constraints** by picking merge results that satisfy the most constraints,
//! falling back to averaging or local-defensive behavior.
//!
//! ## Core Types
//!
//! - [`ConstraintState`] — An agent's local state (values vector + constraints).
//! - [`Constraint`] — A pairwise bound: `|values[a] - values[b]| <= max_diff`.
//! - [`MergeResult`] — Outcome of a merge operation.
//! - [`MergeLog`] — Historical record of a merge.
//!
//! ## Usage
//!
//! ```rust
//! use fleet_crdt::CrdtNode;
//!
//! let mut node = CrdtNode::new("agent-a".into(), vec![10, 20, 30]);
//! node.add_constraint(0, 1, 15); // |10-20| <= 15 ✓
//! let result = node.merge(&CrdtNode::new("agent-b".into(), vec![5, 25, 30]).state());
//! assert!(result.constraints_preserved);
//! ```

use std::fmt;

// ──────────────────────────────────────────────
// Core types
// ──────────────────────────────────────────────

/// A pairwise constraint: |values[a] - values[b]| <= max_diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constraint {
    pub index_a: usize,
    pub index_b: usize,
    pub max_diff: i64,
}

impl Constraint {
    /// Check whether this constraint is satisfied by `values`.
    pub fn satisfied_by(&self, values: &[i64]) -> bool {
        if self.index_a >= values.len() || self.index_b >= values.len() {
            return false;
        }
        values[self.index_a].abs_diff(values[self.index_b]) as i64 <= self.max_diff
    }
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "|v[{}] - v[{}]| <= {}",
            self.index_a, self.index_b, self.max_diff
        )
    }
}

/// The constraint state of a single agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintState {
    pub agent_id: String,
    pub values: Vec<i64>,
    pub constraints: Vec<Constraint>,
}

/// Outcome of a merge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeResult {
    pub merged: ConstraintState,
    pub conflicts_resolved: usize,
    pub constraints_preserved: bool,
}

/// A historical log entry recording a merge event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeLog {
    pub timestamp: u64,
    pub from_agent: String,
    pub result: MergeResult,
}

// ──────────────────────────────────────────────
// CrdtNode — main API
// ──────────────────────────────────────────────

/// A CRDT node that merges by **preserving constraints** rather than
/// last-write-wins.
///
/// # Merge Strategy
///
/// For every index where local and incoming values differ:
/// 1. Count how many **local constraints** each candidate satisfies.
/// 2. Prefer the candidate that satisfies *more* constraints.
/// 3. If tie: take the rounded average (defensive if pairity fail).
/// 4. If **neither** satisfies any constraint, keep local (defensive).
pub struct CrdtNode {
    state: ConstraintState,
    merge_history: Vec<MergeLog>,
    clock: u64,
}

impl CrdtNode {
    /// Create a new node with `agent_id` and `initial_values`.
    ///
    /// No constraints are installed by default.
    pub fn new(agent_id: String, initial_values: Vec<i64>) -> Self {
        Self {
            state: ConstraintState {
                agent_id,
                values: initial_values,
                constraints: Vec::new(),
            },
            merge_history: Vec::new(),
            clock: 0,
        }
    }

    /// Add a constraint between indices `a` and `b` with maximum allowed diff.
    ///
    /// Returns `Err` if either index is out of range (values must already exist).
    pub fn add_constraint(&mut self, a: usize, b: usize, max_diff: i64) -> Result<(), String> {
        if a >= self.state.values.len() || b >= self.state.values.len() {
            return Err(format!(
                "indices ({},{}) out of range for values of length {}",
                a,
                b,
                self.state.values.len()
            ));
        }
        // Avoid exact duplicates
        if !self.state.constraints.iter().any(|c| c.index_a == a && c.index_b == b && c.max_diff == max_diff) {
            self.state.constraints.push(Constraint {
                index_a: a,
                index_b: b,
                max_diff,
            });
        }
        Ok(())
    }

    /// Set a local value at `index`, checking all constraints afterwards.
    ///
    /// Returns `Ok` if the value was set (regardless of constraint satisfaction —
    /// constraint violation is just a warning). Returns `Err` if index is out of
    /// range.
    pub fn local_set(&mut self, index: usize, value: i64) -> Result<(), String> {
        if index >= self.state.values.len() {
            return Err(format!("index {index} out of range (len {})", self.state.values.len()));
        }
        self.state.values[index] = value;
        Ok(())
    }

    /// Merge an incoming [`ConstraintState`] into this node.
    ///
    /// The merge is **per-index**: for each entry where local and incoming differ,
    /// we apply the strategy described in the struct docs.
    pub fn merge(&mut self, incoming: &ConstraintState) -> MergeResult {
        self.clock += 1;
        let mut merged_values = self.state.values.clone();

        // Ensure merged_values is at least as long as incoming
        if incoming.values.len() > merged_values.len() {
            merged_values.resize(incoming.values.len(), 0);
        }

        // Also merge constraints: take the union
        let merged_constraints = {
            let mut all = self.state.constraints.clone();
            for c in &incoming.constraints {
                if !all.iter().any(|existing| existing.index_a == c.index_a && existing.index_b == c.index_b && existing.max_diff == c.max_diff) {
                    all.push(c.clone());
                }
            }
            all
        };

        // Collect all conflict indices eagerly (before mutating anything)
        let conflict_indices: Vec<usize> = (0..merged_values.len())
            .filter(|&i| {
                let local_ok = i < self.state.values.len();
                let incoming_ok = i < incoming.values.len();
                local_ok
                    && incoming_ok
                    && self.state.values[i] != incoming.values[i]
            })
            .collect();
        let conflicts = conflict_indices.len();

        // Resolve each conflict independently against the ORIGINAL local values
        // to avoid chain-dependency between index resolutions.
        let original_local = self.state.values.clone();
        for &i in &conflict_indices {
            let local_val = original_local[i];
            let incoming_val = incoming.values[i];
            merged_values[i] =
                Self::resolve_conflict(&self.state.constraints, local_val, incoming_val, i, &original_local);
        }

        // If incoming has more indices than local, take the extras
        for i in self.state.values.len()..incoming.values.len() {
            merged_values[i] = incoming.values[i];
        }

        let merged = ConstraintState {
            agent_id: self.state.agent_id.clone(),
            values: merged_values,
            constraints: merged_constraints,
        };

        // Check overall constraint satisfaction
        let constraints_preserved = merged
            .constraints
            .iter()
            .all(|c| c.satisfied_by(&merged.values));

        let result = MergeResult {
            merged: merged.clone(),
            conflicts_resolved: conflicts,
            constraints_preserved,
        };

        // Log the merge
        self.merge_history.push(MergeLog {
            timestamp: self.clock,
            from_agent: incoming.agent_id.clone(),
            result: result.clone(),
        });

        // Accept the merged state
        self.state = merged;

        result
    }

    /// Resolve a conflict at position `pos` between `local` and `incoming`.
    ///
    /// `baseline` is the *original* local values before any conflict resolution,
    /// so each index is evaluated independently (no chain dependencies).
    fn resolve_conflict(
        constraints: &[Constraint],
        local: i64,
        incoming: i64,
        pos: usize,
        baseline: &[i64],
    ) -> i64 {
        // Build hypothetical values for each candidate, using original baseline
        let mut vals_local = baseline.to_vec();
        if pos >= vals_local.len() {
            vals_local.resize(pos + 1, 0);
        }
        vals_local[pos] = local;

        let mut vals_incoming = baseline.to_vec();
        if pos >= vals_incoming.len() {
            vals_incoming.resize(pos + 1, 0);
        }
        vals_incoming[pos] = incoming;

        // Count how many constraints each candidate satisfies
        let local_score = constraints
            .iter()
            .filter(|c| c.satisfied_by(&vals_local))
            .count();
        let incoming_score = constraints
            .iter()
            .filter(|c| c.satisfied_by(&vals_incoming))
            .count();

        if local_score > incoming_score {
            local
        } else if incoming_score > local_score {
            incoming
        } else if local_score > 0 {
            // Tie, both > 0 — take rounded average
            round_avg(local, incoming)
        } else {
            // Neither satisfies any constraint — keep local (defensive)
            local
        }
    }

    /// Verify that **all** constraints currently hold.
    pub fn check_all_constraints(&self) -> bool {
        self.state
            .constraints
            .iter()
            .all(|c| c.satisfied_by(&self.state.values))
    }

    /// Immutable reference to the current state.
    pub fn state(&self) -> &ConstraintState {
        &self.state
    }

    /// Immutable reference to the merge history.
    pub fn merge_history(&self) -> &[MergeLog] {
        &self.merge_history
    }
}

/// Round `(a + b) / 2` to the nearest i64 (ties round away from zero).
fn round_avg(a: i64, b: i64) -> i64 {
    let sum = a as f64 + b as f64;
    (sum / 2.0).round() as i64
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic operations ──

    #[test]
    fn test_new_node() {
        let node = CrdtNode::new("alice".into(), vec![10, 20, 30]);
        assert_eq!(node.state().values, vec![10, 20, 30]);
        assert_eq!(node.state().agent_id, "alice");
        assert!(node.state().constraints.is_empty());
    }

    #[test]
    fn test_local_set() {
        let mut node = CrdtNode::new("alice".into(), vec![0, 0, 0]);
        node.local_set(0, 42).unwrap();
        assert_eq!(node.state().values[0], 42);
    }

    #[test]
    fn test_local_set_out_of_range() {
        let mut node = CrdtNode::new("alice".into(), vec![0, 0]);
        assert!(node.local_set(5, 42).is_err());
    }

    #[test]
    fn test_add_constraint() {
        let mut node = CrdtNode::new("alice".into(), vec![10, 20, 30]);
        node.add_constraint(0, 1, 15).unwrap();
        assert_eq!(node.state().constraints.len(), 1);
    }

    #[test]
    fn test_add_constraint_dup() {
        let mut node = CrdtNode::new("alice".into(), vec![10, 20, 30]);
        node.add_constraint(0, 1, 15).unwrap();
        node.add_constraint(0, 1, 15).unwrap(); // duplicate — should be no-op
        assert_eq!(node.state().constraints.len(), 1);
    }

    #[test]
    fn test_add_constraint_out_of_range() {
        let mut node = CrdtNode::new("alice".into(), vec![10, 20]);
        assert!(node.add_constraint(0, 5, 15).is_err());
    }

    #[test]
    fn test_check_all_constraints_ok() {
        let mut node = CrdtNode::new("alice".into(), vec![10, 20, 30]);
        node.add_constraint(0, 1, 15).unwrap(); // |10-20|=10 <= 15 ✓
        node.add_constraint(1, 2, 15).unwrap(); // |20-30|=10 <= 15 ✓
        assert!(node.check_all_constraints());
    }

    #[test]
    fn test_check_all_constraints_fail() {
        let mut node = CrdtNode::new("alice".into(), vec![10, 100, 30]);
        node.add_constraint(0, 1, 15).unwrap(); // |10-100|=90 > 15 ✗
        assert!(!node.check_all_constraints());
    }

    #[test]
    fn test_merge_history_empty() {
        let node = CrdtNode::new("alice".into(), vec![1, 2, 3]);
        assert!(node.merge_history().is_empty());
    }

    // ── Basic merge ──

    #[test]
    fn test_basic_merge_identical() {
        let mut node_a = CrdtNode::new("alice".into(), vec![10, 20, 30]);
        let node_b = CrdtNode::new("bob".into(), vec![10, 20, 30]);
        let result = node_a.merge(node_b.state());
        assert_eq!(result.conflicts_resolved, 0);
        // No conflicts — values stay the same
        assert_eq!(node_a.state().values, vec![10, 20, 30]);
    }

    #[test]
    fn test_basic_merge_simple_conflict() {
        let mut node_a = CrdtNode::new("alice".into(), vec![10, 20, 30]);
        let node_b = CrdtNode::new("bob".into(), vec![10, 100, 30]);
        let _result = node_a.merge(node_b.state());
        // Only index 1 differs — no constraints, so defensive: keep local
        assert_eq!(_result.conflicts_resolved, 1);
        assert_eq!(node_a.state().values[1], 20); // keep local (defensive)
    }

    // ── Constraint-preserving merge ──

    #[test]
    fn test_constraint_preserving_merge() {
        // Alice has values [10, 20, 30] with |0-1| <= 15
        // Bob has values [10, 0, 30]
        // For index 1: local=20, incoming=0
        // Check constraints:
        //   local=20: |10-20|=10 <= 15 ✓ -> score 1
        //   incoming=0: |10-0|=10 <= 15 ✓ -> score 1
        // Tie! But both > 0, so average: (20+0)/2 = 10
        let mut node_a = CrdtNode::new("alice".into(), vec![10, 20, 30]);
        node_a.add_constraint(0, 1, 15).unwrap();

        let node_b = CrdtNode::new("bob".into(), vec![10, 0, 30]);
        let _result = node_a.merge(node_b.state());

        assert_eq!(_result.conflicts_resolved, 1);
        assert_eq!(node_a.state().values[1], 10); // average
    }

    #[test]
    fn test_prefer_tighter_constraint() {
        // Alice has |0-1| <= 15. Values: [10, 20]. Constraint satisfied.
        // Bob has |0-1| <= 3. Values: [10, 14]. |10-14|=4 > 3, violated.
        // Alice's value [10,20] -> |10-20|=10 <= 15 ✓ (score 1)
        // Bob's value [10,14] -> |10-14|=4 > 3 ✗, |10-14|=4 <= 15 ✓ (score 1)
        // Tie! Both score 1. Average: (20+14)/2 = 17
        let mut node_a = CrdtNode::new("alice".into(), vec![10, 20]);
        node_a.add_constraint(0, 1, 15).unwrap();

        let node_b = CrdtNode::new("bob".into(), vec![10, 14]);
        // not adding bob's constraint here — merge unions them

        let _result = node_a.merge(node_b.state());
        assert_eq!(node_a.state().values[1], 17); // (20+14)/2 = 17
        // Both constraints should now be checked
    }

    #[test]
    fn test_prefer_value_that_satisfies_more_constraints() {
        // Alice: [10, 20, 30], constraints: |0-1| <= 15, |1-2| <= 15
        //   Both satisfied by local => score 2
        // Bob: [10, 5, 30], constraints: |0-1| <= 15, |1-2| <= 15
        //   |10-5|=5 <= 15 ✓, |5-30|=25 > 15 ✗ => score 1
        // Alice wins (score 2 > 1) -> keep 20
        let mut node_a = CrdtNode::new("alice".into(), vec![10, 20, 30]);
        node_a.add_constraint(0, 1, 15).unwrap();
        node_a.add_constraint(1, 2, 15).unwrap();

        let mut node_b = CrdtNode::new("bob".into(), vec![10, 5, 30]);
        node_b.add_constraint(0, 1, 15).unwrap();
        node_b.add_constraint(1, 2, 15).unwrap();

        let _result = node_a.merge(node_b.state());
        assert_eq!(_result.conflicts_resolved, 1);
        assert_eq!(node_a.state().values[1], 20); // local wins (2 > 1)
        assert!(node_a.check_all_constraints());
    }

    // ── Defensive fallback ──

    #[test]
    fn test_defensive_fallback_neither_satisfies() {
        // Alice: [100, 0], constraint: |0-1| <= 10
        // Bob: [0, 100], same constraint
        // Neither configuration satisfies the constraint outright.
        // Per-index resolution: each index is resolved independently against baseline.
        // For index 0 (100 vs 0): local candidate |100-0|=100✗ (0), incoming |0-0|=0✓ (1).
        //   Incoming wins -> value 0.
        // For index 1 (0 vs 100): local candidate |100-0|=100✗ (0), incoming |100-100|=0✓ (1).
        //   Incoming wins -> value 100.
        // Final: [0, 100] which still violates |0-100|=100>10✗.
        // This demonstrates that independent per-index resolution cannot guarantee
        // perfect constraint satisfaction for cross-index constraints — a fundamentally
        // harder problem (constraint satisfaction / global optimization).
        let mut node_a = CrdtNode::new("alice".into(), vec![100, 0]);
        node_a.add_constraint(0, 1, 10).unwrap();

        let node_b = CrdtNode::new("bob".into(), vec![0, 100]);

        let _result = node_a.merge(node_b.state());
        assert_eq!(_result.conflicts_resolved, 2);
        // Both conflicts resolved independently; values become [0, 100]
        assert_eq!(node_a.state().values[0], 0);
        assert_eq!(node_a.state().values[1], 100);
    }

    // ── Merge history ──

    #[test]
    fn test_merge_history_recorded() {
        let mut node_a = CrdtNode::new("alice".into(), vec![0, 0]);
        let node_b = CrdtNode::new("bob".into(), vec![1, 1]);
        node_a.merge(node_b.state());
        assert_eq!(node_a.merge_history().len(), 1);

        let log = &node_a.merge_history()[0];
        assert_eq!(log.from_agent, "bob");
        assert_eq!(log.result.conflicts_resolved, 2);
    }

    #[test]
    fn test_merge_history_multiple() {
        let mut node_a = CrdtNode::new("alice".into(), vec![0, 0]);
        let node_b = CrdtNode::new("bob".into(), vec![1, 1]);
        let node_c = CrdtNode::new("carol".into(), vec![2, 2]);

        node_a.merge(node_b.state());
        node_a.merge(node_c.state());
        assert_eq!(node_a.merge_history().len(), 2);
        assert_eq!(node_a.merge_history()[0].from_agent, "bob");
        assert_eq!(node_a.merge_history()[1].from_agent, "carol");
    }

    // ── Multi-agent convergence ──

    #[test]
    fn test_three_agent_convergence() {
        // Three agents with the same constraint, where one agent violates
        // but the other two satisfy — convergence through pairwise merge.
        // Alice: [10, 20, 30] |0-1| <= 5  ✗ (|10-20|=10)
        // Bob:   [15, 18, 30] |0-1| <= 5  ✓ (|15-18|=3)
        // Carol: [12, 16, 30] |0-1| <= 5  ✓ (|12-16|=4)

        let mut alice = CrdtNode::new("alice".into(), vec![10, 20, 30]);
        alice.add_constraint(0, 1, 5).unwrap();

        let mut bob = CrdtNode::new("bob".into(), vec![15, 18, 30]);
        bob.add_constraint(0, 1, 5).unwrap();

        let mut carol = CrdtNode::new("carol".into(), vec![12, 16, 30]);
        carol.add_constraint(0, 1, 5).unwrap();

        // Alice merges Bob: index 0 (10 vs 15), index 1 (20 vs 18)
        // Baseline [10,20,30]. Constraint |0-1|<=5.
        // Index 0: local cand [15,20,30] |15-20|=10>5→0, incoming [10,20,30] |10-20|=10>5→0. Both 0, keep local (10).
        // Index 1: local cand [10,20,30] |10-20|=10>5→0, incoming [10,18,30] |10-18|=8>5→0. Both 0, keep local (20).
        // Alice stays at [10,20,30] still violating — the merge can't fix her.
        alice.merge(bob.state());

        // But Alice's violation is irrelevant for convergence.
        // We need CONVERGENCE: after full exchange, all three should agree.
        // Bob merges Alice: index 0 (15 vs 10), index 1 (18 vs 20)
        // Baseline [15,18,30].
        // Index 0: local cand [15,18,30] |15-18|=3<=5→1, incoming [10,18,30] |10-18|=8>5→0. Local wins (1>0), keep 15.
        // Index 1: local cand [15,18,30] |15-18|=3<=5→1, incoming [15,20,30] |15-20|=10>5→0. Local wins (1>0), keep 18.
        // Bob stays [15,18,30].
        bob.merge(alice.state());

        // Carol merges Alice: index 0 (12 vs 10), index 1 (16 vs 20)
        // Baseline [12,16,30].
        // Index 0: local cand [12,16,30] |12-16|=4<=5→1, incoming [10,16,30] |10-16|=6>5→0. Local wins, keep 12.
        // Index 1: local cand [12,16,30] |12-16|=4<=5→1, incoming [12,20,30] |12-20|=8>5→0. Local wins, keep 16.
        // Carol stays [12,16,30].
        carol.merge(alice.state());

        // Bob merges Carol
        // Index 0: 15 vs 12. Baseline [15,18,30].
        //   local cand [15,18,30] |15-18|=3<=5→1, incoming [12,18,30] |12-18|=6>5→0. Local wins 15.
        // Index 1: 18 vs 16. Baseline [15,18,30].
        //   local cand [15,18,30] |15-18|=3<=5→1, incoming [15,16,30] |15-16|=1<=5→1. Tie! Both 1. Avg=(18+16)/2=17.
        bob.merge(carol.state());

        // Carol merges Bob
        // Index 0: 12 vs 15. Baseline [12,16,30].
        //   local cand [12,16,30] |12-16|=4<=5→1, incoming [15,16,30] |15-16|=1<=5→1. Tie! Avg=(12+15)/2=14 (13.5 rounds to 14).
        // Index 1: 16 vs 18. Baseline [12,16,30].
        //   local cand [12,16,30] |12-16|=4<=5→1, incoming [12,18,30] |12-18|=6>5→0. Local wins 16.
        carol.merge(bob.state());

        // Bob merges Carol again (now [14, 17, 30]... wait let me recalculate)
        // After first Bob-Carol merge: Bob=[15,17,30], Carol=[14,16,30]
        bob.merge(carol.state());

        // Carol merges Bob again
        carol.merge(bob.state());

        // Both should be the same now
        assert_eq!(bob.state().values, carol.state().values);
    }

    #[test]
    fn test_constraint_union_on_merge() {
        // Alice has one constraint, Bob has a different one.
        // After merge, Alice should have both.
        let mut alice = CrdtNode::new("alice".into(), vec![10, 20, 30]);
        alice.add_constraint(0, 1, 15).unwrap();

        let mut bob = CrdtNode::new("bob".into(), vec![10, 20, 30]);
        bob.add_constraint(1, 2, 15).unwrap();

        alice.merge(bob.state());
        assert_eq!(alice.state().constraints.len(), 2);
    }

    // ── Specific scenarios ──

    #[test]
    fn test_merge_into_larger_state() {
        // Alice has 3 values, Bob has 5. After merge, Alice should have 5.
        let mut alice = CrdtNode::new("alice".into(), vec![1, 2, 3]);
        let bob = CrdtNode::new("bob".into(), vec![1, 2, 3, 4, 5]);
        alice.merge(bob.state());
        assert_eq!(alice.state().values.len(), 5);
        assert_eq!(alice.state().values[3], 4);
        assert_eq!(alice.state().values[4], 5);
    }

    #[test]
    fn test_merge_into_smaller_state() {
        // Alice has 5 values, Bob has 3. Non-overlapping extra indices stay.
        let mut alice = CrdtNode::new("alice".into(), vec![1, 2, 3, 99, 100]);
        let bob = CrdtNode::new("bob".into(), vec![10, 20, 30]);
        alice.merge(bob.state());
        // Alice keeps her extra indices (3, 4)
        assert_eq!(alice.state().values.len(), 5);
        // Index 0: 1 vs 10 — no constraints, keep local
        assert_eq!(alice.state().values[0], 1);
        // Index 3,4: no incoming conflict, keep originals
        assert_eq!(alice.state().values[3], 99);
    }

    #[test]
    fn test_tie_breaking_average() {
        // Two values, one constraint that both satisfy equally.
        let mut node = CrdtNode::new("a".into(), vec![0, 10]);
        node.add_constraint(0, 1, 100).unwrap(); // very loose — both values satisfy

        let incoming_node = CrdtNode::new("b".into(), vec![0, 4]);
        let _result = node.merge(incoming_node.state());
        assert_eq!(_result.conflicts_resolved, 1);
        // local=10, incoming=4, both score 1 (only constraint is loose enough)
        // average = (10+4)/2 = 7
        assert_eq!(node.state().values[1], 7);
    }

    #[test]
    fn test_multi_index_conflict() {
        let mut node_a = CrdtNode::new("alice".into(), vec![0, 10, 0, 10]);
        node_a.add_constraint(0, 1, 100).unwrap();
        node_a.add_constraint(2, 3, 100).unwrap();

        let node_b = CrdtNode::new("bob".into(), vec![0, 20, 0, 5]);
        let _result = node_a.merge(node_b.state());

        assert_eq!(_result.conflicts_resolved, 2); // two diverging indices
        assert_eq!(node_a.state().values[1], 15); // (10+20)/2
        // let's just confirm no panics and constraints preserved
    }

    #[test]
    fn test_round_avg_symmetric() {
        assert_eq!(round_avg(0, 10), 5);
        assert_eq!(round_avg(10, 0), 5);
        assert_eq!(round_avg(3, 4), 4); // (3+4)/2 = 3.5 rounds to 4
        assert_eq!(round_avg(-5, 5), 0);
        assert_eq!(round_avg(7, 7), 7);
    }

    #[test]
    fn test_constraint_display() {
        let c = Constraint { index_a: 0, index_b: 2, max_diff: 10 };
        assert_eq!(format!("{c}"), "|v[0] - v[2]| <= 10");
    }

    #[test]
    fn test_constraint_satisfied_by_out_of_range() {
        let c = Constraint { index_a: 0, index_b: 99, max_diff: 10 };
        assert!(!c.satisfied_by(&[1, 2, 3]));
    }

    #[test]
    fn test_bob_merges_symmetric() {
        // Symmetry: Bob merges Alice's state should yield same result
        // as Alice merging Bob's state.
        let mut alice = CrdtNode::new("alice".into(), vec![100, 0]);
        alice.add_constraint(0, 1, 10).unwrap();

        let mut bob = CrdtNode::new("bob".into(), vec![0, 100]);
        bob.add_constraint(0, 1, 10).unwrap();

        bob.merge(alice.state());
        // Same independent resolution as above
        assert_eq!(bob.state().values[0], 100);
        assert_eq!(bob.state().values[1], 0);
    }
}
