---
hide:
  - navigation
---

--8<-- "README.md"

## Documentation Map

- [Results Snapshot](results_snapshot.md): current evidence ledger, expanded data
  gates, benchmark smoke outputs, and paper-readiness boundaries.
- [Paper Plan](paper_plan.md): research question, contribution target, execution
  gates, claim ladder, and paper-facing boundaries.
- [Data](data.md): source roles, option identifiers, timestamp policy, expanded
  silver artifacts, IV-inversion assumptions, and U.S./Japan OOD data gates.
- [Benchmark Protocol](benchmark_protocol.md): fixed splits, masked
  reconstruction, train-only baselines, leakage controls, regularizers, seeds,
  and acceptance rules.
- [Graph](graph.md): option-token nodes, edge families, liquidity weights,
  masking semantics, and decoded-price no-arbitrage hooks.
- [Development Audit Prompt](audit/development_audit_prompt.md): code and data
  review prompt before implementation handoffs.
- [Manuscript Audit Prompt](audit/manuscript_audit_prompt.md): claim and
  reviewer-facing audit prompt before draft circulation.
- [Future Work](future_work.md): deferred extensions that should not dilute the
  current paper.

## Reading Order

1. Start with the [Results Snapshot](results_snapshot.md) to see what evidence
   currently exists.
2. Use the [Paper Plan](paper_plan.md) as the research contract.
3. Read [Data](data.md) before changing ingestion, IV inversion, or OOD scope.
4. Read [Benchmark Protocol](benchmark_protocol.md) before changing train/test
   splits, baseline definitions, or paper-facing metrics.
5. Read [Graph](graph.md) before changing node features, edge construction, or
   no-arbitrage hooks.
6. Use the audit prompts before implementation review or manuscript review.
