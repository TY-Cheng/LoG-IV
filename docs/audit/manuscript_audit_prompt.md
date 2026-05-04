# Manuscript Audit Prompt

Use this prompt before turning LoG-IV outputs into paper text.

## Reader Lens

Review as a LoG or graph-learning reviewer with finance-domain skepticism. The
paper must make clear why the graph geometry matters and why the finance
constraints are more than decoration.

## Claim Checks

1. Is the option chain described as an irregular set or graph, not forced into a
   fixed image without justification?
2. Are liquidity, spread, volume, and open interest used as modeling information
   or diagnostics rather than post-hoc explanations?
3. Are no-arbitrage claims supported by explicit violation metrics?
4. Are U.S. in-domain, held-out-underlying, and Japan out-of-distribution
   results separated?
5. Are Japanese data limitations described as domain-shift evidence rather than
   brushed aside?
6. Are baselines strong enough for a graph-learning venue?
7. Are trading, hedging, or execution claims excluded unless directly tested?
8. Is every table or figure tied to a reproducible run manifest?

## Output Format

List likely rejection paths first, then fixable issues, then wording or
positioning improvements. Keep the review grounded in the actual evidence files.
