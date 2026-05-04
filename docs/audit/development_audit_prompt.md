# Development Audit Prompt

Use this prompt before accepting implementation work in LoG-IV.

## Scope

Review the code as an option-surface graph learning repository. Separate:

- implemented behavior;
- placeholder behavior;
- planned research design;
- empirical claims that are not yet supported.

## Checks

1. Does the code preserve the canonical option-token key:
   `(market, underlying, observation_date, expiry, strike, option_type)`?
2. Are timestamp fields explicit enough to prevent mixing observation time,
   vendor availability, model cutoff, and target time?
3. Are liquidity fields such as spread, volume, and open interest preserved
   before smoothing or message passing?
4. Does graph construction record edge type, direction, and weight?
5. Are no-arbitrage checks labeled as diagnostics unless they are actually
   computed on registered surfaces?
6. Do data-fetching routines keep secrets and licensed payloads out of git?
7. Do tests cover schema validation and at least one graph fixture?
8. Does documentation match the current implementation rather than the intended
   final model?

## Output Format

Lead with bugs, leakage risks, misleading claims, missing tests, or stale docs.
Use file and line references for each finding. If no blocking issue is found,
state the remaining empirical and implementation gaps explicitly.
