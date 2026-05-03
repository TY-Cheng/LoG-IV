# Paper Plan

## Working Title

**Liquidity-Aware Graph Neural Operators for Irregular Option Surfaces**

## Core Question

Real option chains are irregular point clouds, not fixed images. Maturity grids
are uneven, strike grids differ by expiry, and quote quality depends on bid-ask
spread, volume, open interest, and market-specific contract conventions.

This project asks whether graph or token operators that preserve option-surface
geometry and liquidity state can improve masked option-surface reconstruction
against credible train-only interpolation baselines.

## Current Paper Scope

The first paper should stay centered on a U.S. end-of-day option-chain benchmark.
Japan is useful as an OOD probe after its data gate passes, but the first claim
does not depend on Japan transfer.

In scope:

- U.S. option-token graphs built from expanded Massive OPRA daily aggregates.
- Same-day stock daily closes as the first spot/forward proxy for inferred IV.
- Masked reconstruction as the headline task.
- Fixed temporal, ticker-holdout, and temporal-ticker split protocols.
- Train-only non-neural baselines, especially moneyness-tenor kNN.
- Decoded-price diagnostics and decoded-price calendar/convexity regularizers.
- Japan option data as OOD coverage and domain-shift evidence, not as a required
  positive transfer result.

Out of scope for the first paper:

- Trading PnL, hedging, execution-cost, or live risk-monitoring claims.
- Intraday option graphs.
- Vendor-IV claims for Massive day aggregates.
- No-arbitrage claims that depend on unstated rate, dividend, or forward curves.
- Japan transfer claims unless matched fixed baselines and market assumptions are
  documented.

## Claim Ladder

| Gate | Claim status | Requirement |
| --- | --- | --- |
| Engineering smoke | Code path works | Tests, strict docs build, source probes, and toy/real smoke artifacts pass. |
| Data gate | Benchmark may run | U.S. has at least 1,000 usable surfaces and Japan has at least 20 usable observation dates for OOD probing. |
| Protocol gate | Results are credible diagnostics | Fixed splits, masked reconstruction, train-only baselines, leakage diagnostics separated, and decoded-price regularizer semantics recorded. |
| Benchmark gate | Paper-candidate evidence | Multi-seed longer training beats credible kNN/interpolation baselines on headline masked metrics without worse diagnostics. |
| Claim gate | Manuscript claim | Liquidity/no-arbitrage/OOD diagnostics support the interpretation, not only the scalar loss. |

No result should be promoted to `paper_candidate` before the benchmark gate.

## LoG Venue Strategy

LoG is a natural venue only if the manuscript is framed as a graph and geometry
learning paper, not as a finance application paper. The paper should sell the
irregular option chain as a graph learning problem: non-uniform strike grids,
non-uniform maturity grids, liquidity-dependent noise, and decoded-price
no-arbitrage diagnostics.

The current project is an applied graph ML and benchmark paper. It is not a
theory paper unless we add formal results on operator stability, liquidity-
weighted smoothing, no-arbitrage projection, or a related graph-signal model.

Submission strategy:

- Target LoG proceedings only if the benchmark gate passes with multi-seed
  evidence, credible non-neural and neural baselines, and complete artifacts.
- Use a non-archival or extended abstract route if final results are mixed,
  preliminary, or best positioned as a benchmark/protocol contribution.
- Do not write claims about trading performance, market superiority, or live
  risk management.
- Keep Japan as a graph-domain-shift probe until market-specific normalization,
  timestamp semantics, rates, dividends, and forward assumptions are documented.

Relevant LoG scope notes:

- LoG 2025 included proceedings and non-archival extended abstract tracks.
- LoG 2025 subject areas included GNN architectures, graph signal processing,
  self-supervised learning, and graph/geometric ML datasets, benchmarks, and
  infrastructure.
- LoG 2026 CFP details should be checked again before submission because the
  current official CFP page may change.

Sources:

- <https://logconference.org/cfp/>
- <https://log2025.logconference.org/cfp/>

## Current Evidence Boundary

The expanded data gate is much stronger than the initial MVP:

- U.S. expanded silver data: 1,516,612 rows, 31 observation dates, 40
  underlyings, 1,240 surfaces, and 1,240 surfaces with at least 20 nodes.
- Japan expanded silver data: 531,280 rows, 31 observation dates, 249
  underlying or surface keys, 6,991 surfaces, and 6,179 surfaces with at least
  20 nodes.

This is enough for a first LoG masked-reconstruction benchmark, but not enough
for strong temporal generalization, regime robustness, or cross-market transfer
claims. The main limitation is time coverage: 31 observation dates is a credible
first panel, not a market-cycle study.

The current `benchmark_stage1` run is promising but still diagnostic:

- It uses 10 epochs and one seed.
- GNN variants beat the current train-only moneyness-tenor kNN baseline on
  masked U.S. IV MAE in that run.
- Decoded regularization improves some calendar diagnostics, but convexity
  violations remain high enough that no-arbitrage claims are not yet mature.

Do not use these results as final LoG evidence without longer multi-seed runs
and stronger baselines.

## Headline Task

The first headline task is **masked reconstruction**:

- The target is IV at observed option contracts.
- Masked input nodes remove IV, bid, ask, volume, and open interest from model
  inputs.
- Target values remain available only to the loss and metrics.
- Headline metrics are computed on masked nodes, not on all observed nodes.

This is the right first task because it tests whether the representation uses
neighboring contracts, liquidity state, and surface geometry without requiring a
clean next-day forecasting target.

## Baselines

P0 train-only baselines:

- train global mean IV;
- train mean IV by underlying;
- train mean IV by moneyness-tenor bucket;
- train-fitted moneyness-tenor kNN;
- random uniform IV.

P0 within-surface baselines may use evaluation-surface visible nodes after the
mask is applied:

- within-surface kNN;
- within-surface RBF interpolation;
- within-surface local linear interpolation;
- raw SVI per-expiry fitting with timeout and failure accounting.

Leave-one-out baselines may be written only as leakage-prone diagnostics. They
must not be used as paper-facing benchmark comparisons.

The current train-only kNN baseline is a valid credibility floor, but it is not
sufficient for a LoG paper. SSVI, constrained SVI, and the heavier ML/graph
baseline family are P1 until `data_v1` passes:

- constrained SVI and SSVI calendar fitting;
- graph Laplacian smoothing;
- tabular LightGBM or XGBoost on geometry and liquidity features;
- DeepSets or Set Transformer as no-edge neural set baselines;
- vanilla GCN, GAT, and GraphSAGE variants as standard graph baselines.

A graph operator result that beats only global means is not publishable. A
publishable result should beat the strongest train-only interpolation or
surface-fitting baseline on masked IV MAE or normalized decoded-price MAE.

## Model Hypotheses

- Liquidity-aware message passing should avoid over-propagating stale, wide, or
  low-volume quotes.
- Graph operators should handle irregular strike and tenor grids more naturally
  than fixed-grid image models.
- Decoded-price regularization should reduce financially implausible surfaces
  without relying on embedding-distance proxies.

These hypotheses become paper claims only if they survive the benchmark gate.

## Metrics

Headline metrics:

- masked IV MAE;
- masked IV RMSE;
- model delta versus train-only kNN;
- liquidity-stratified masked IV error.

Supporting diagnostics:

- normalized decoded-price error;
- calendar and strike-convexity violation counts on true-IV and predicted-IV
  decoded prices;
- OOD degradation from U.S. to Japan or held-out tickers;
- sensitivity to seed and training length.

## Reviewer Risk Register

Likely LoG reviewer concerns:

- The paper may look like a finance application unless the graph learning
  benchmark and operator contribution are foregrounded.
- The data window may be considered too short for generalization claims.
- The baseline set may be considered weak without SVI/SSVI, interpolation,
  tabular ML, set models, and standard GNN baselines.
- The model may be considered insufficiently SOTA if compared only with simple
  means and kNN.
- Decoded-price diagnostics may be questioned unless forward, rate, dividend,
  and IV inversion assumptions are explicit.
- Japan OOD claims may be rejected if the market normalization and timestamp
  semantics are not documented.
- No-arbitrage claims may be rejected if convexity or calendar diagnostics are
  worse than strong baselines.

Mitigations:

- Keep the headline claim to leakage-controlled masked reconstruction of
  irregular option-surface graphs.
- Report fixed split manifests, train-only baseline fitting, and masked-node
  metrics as first-class artifacts.
- Run 100-epoch, 3-seed or larger experiments before final tables.
- Add the stronger baseline ladder before claiming graph superiority.
- Report liquidity, moneyness, tenor, option-type, ticker, and date-stratified
  metrics.
- Label Japan results as `japan_ood_probe` until transfer evidence is strong.

## Expected Sell

The strongest paper positioning is:

> We introduce a leakage-controlled masked-reconstruction benchmark for
> irregular option-surface graphs, and evaluate liquidity-aware graph neural
> operators against train-only interpolation, surface-fitting, tabular, set, and
> graph baselines on real U.S. options, with Japan as an explicit graph-domain
> shift probe.

This sell is viable only if the final benchmark shows that the graph operator
beats the strongest train-only benchmark on masked U.S. test error and does not
materially worsen decoded-price no-arbitrage diagnostics.

## Immediate Execution Plan

1. Keep the expanded U.S./Japan silver data gates green.
2. Add the stronger baseline ladder before paper-candidate runs.
3. Run `benchmark-protocol` with fixed temporal masked reconstruction.
4. Start with capped smoke runs for pipeline checks.
5. Promote only multi-seed longer runs to paper-candidate consideration.
6. Diagnose GNN versus the strongest baseline only after the protocol and data gates are already
   satisfied.
