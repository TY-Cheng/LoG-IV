# Research Plan

## Working Title

**LoG-IV: Leakage-Controlled Graph Learning for Irregular Option-Implied
Volatility Surfaces**

## Core Question

Real option chains are irregular point clouds, not fixed images. Maturity grids
are uneven, strike grids differ by expiry, and quote quality depends on bid-ask
spread, volume, open interest, and market-specific contract conventions.

This project asks whether graph or token models that condition on quote
reliability can improve leakage-controlled masked reconstruction of
option-implied volatility surfaces under realistic missingness. Liquidity
variables are treated as noisy indicators of quote reliability, not merely as
additional node features.

## Core Terminology

In this project, **leakage** means information leakage in the evaluation
protocol. It does not refer to privacy or data-security leakage. In masked
reconstruction, a query node is an option contract whose IV or price-related
target is hidden from the model. A benchmark leaks if the query node, a
normalizer, or a context aggregate exposes information that is unavailable under
the stated prediction task.

The most important leakage channels are quote-derived fields on masked query
nodes: bid, ask, mid, spread, same-day decoded price, same-day IV, and Greeks
computed from same-day IV. Surface-level or neighborhood aggregates can also
leak if they are computed before applying the mask and therefore include target
nodes. LoG-IV therefore requires masked query nodes to carry only geometry,
contract metadata, and explicitly allowed lagged or exogenous variables. All
visible-context aggregates must be computed after masking, using visible nodes
only. Normalization must use train-only statistics or visible-only statistics
under the current mask.

**Heteroscedastic** means that observation noise is not constant across option
quotes. In option markets, liquid near-the-money contracts with narrow spreads
are usually more reliable observations than stale, wide-spread, low-volume, or
far-out-of-the-money contracts. The project models liquidity variables as noisy
evidence about observation precision, not as a direct target and not as a claim
that liquidity mechanically causes volatility.

A simple statistical view is: the observed quote equals a latent surface value
plus observation error, and the variance of that error is quote-specific. The
inverse variance is the quote precision. A heteroscedastic graph model may use
estimated precision to weight message passing, weight the training likelihood,
or both. This is stronger than simply appending spread, volume, and open
interest as ordinary node features, and it must be supported by ablations before
it becomes a manuscript-level claim.

## Current Paper Scope

The first paper should stay centered on a graph-learning benchmark for
irregular option-surface graphs with liquidity-dependent observation noise.
Japan is useful for out-of-distribution evaluation after its data gate passes,
but the first claim does not depend on positive Japan transfer results.

In scope:

- U.S. option-token graphs built from expanded Massive OPRA daily aggregates.
- Same-day stock daily closes as the first spot/forward proxy for inferred IV.
- Masked reconstruction as the main task.
- Fixed temporal, ticker-holdout, and temporal-ticker split protocols.
- Train-only non-neural baselines, especially moneyness-tenor kNN.
- Decoded-price diagnostics and decoded-price calendar/convexity regularizers.
- Synthetic-LoG-IV as a reproducible synthetic benchmark with diagnostic-only
  fields.
- Japan option data as out-of-distribution evidence, not as a required positive
  transfer result.

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
| Engineering check | Code path works | Tests, strict docs build, source probes, and small real-data artifacts pass. |
| Data gate | Benchmark may run | U.S. has at least 2,400 usable surfaces and Japan has at least 20 usable observation dates for out-of-distribution evaluation. |
| Protocol gate | Results are interpretable | Fixed splits, masked reconstruction, train-only baselines, separated leakage diagnostics, and decoded-price regularizer semantics are recorded. |
| Benchmark gate | Manuscript-level evidence | Multi-seed runs beat kNN/interpolation baselines on masked-node metrics without materially worse diagnostics. |
| Claim gate | Manuscript claim | Liquidity, no-arbitrage, and distribution-shift diagnostics support the interpretation, not only the scalar loss. |

No result should be treated as manuscript-level evidence before the benchmark
gate.

## LoG Venue Strategy

LoG is a natural venue only if the manuscript is framed as a graph and geometry
learning paper, not as a finance application paper. The paper should describe
the irregular option chain as a graph learning problem: non-uniform strike
grids, non-uniform maturity grids, liquidity-dependent noise, and decoded-price
surface diagnostics.

The current project is an applied graph ML and benchmark paper. It is not a
theory paper unless we add formal results on operator stability, liquidity-
weighted smoothing, no-arbitrage projection, or a related graph-signal model.

Submission strategy:

- Target LoG proceedings only if the benchmark gate passes with multi-seed
  evidence, non-neural and neural baselines, and complete artifacts.
- Use a non-archival or extended abstract route if final results are mixed,
  preliminary, or best positioned as a benchmark/protocol contribution.
- Do not write claims about trading performance, market superiority, or live
  risk management.
- Keep Japan as an out-of-distribution evaluation setting until market-specific
  normalization, timestamp semantics, rates, dividends, and forward assumptions
  are documented.

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

## Related-Work Anchors

The manuscript should not claim that graph models, neural operators, or neural
smoothing for implied-volatility surfaces are new in themselves. Closely related
work already covers neural IV surface smoothing, graph or operator architectures
for irregular option observations, heterogeneous graph attention for IVS
prediction, real-time arbitrage-free smoothing, and sparse-quote surface
completion.

The closest anchors are:

| Work | Why it is close | Boundary for LoG-IV |
| --- | --- | --- |
| [Ackerer, Tagasovska, and Vatter, **Deep Smoothing of the Implied Volatility Surface**, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/858e47701162578e5e627cd93ab0938a-Abstract.html) | Establishes neural IV surface fitting/prediction with soft no-arbitrage penalties and discusses sparse or erroneous market data. | LoG-IV should not claim neural smoothing or no-arbitrage penalties as new. Its narrower contribution is leakage-controlled graph benchmarking and liquidity-dependent observation noise. |
| [Wiedemann, Jacquier, and Gonon, **Operator Deep Smoothing for Implied Volatility**, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f115f619b62833aadc5acb058975b0e6-Abstract-Conference.html) | Uses a graph neural operator to map irregular observed option data directly to smoothed surfaces; reports ten years of intraday S&P 500 option data, no-arbitrage adherence, subsampling robustness, and SVI comparison. | This is the most direct method anchor. LoG-IV must differentiate through leakage-controlled masking, liquidity-dependent reliability modeling, market-relevant missingness regimes, and broader benchmark artifacts. |
| [Liang et al., **Hexagon-Net: Heterogeneous Cross-View Aligned Graph Attention Networks for Implied Volatility Surface Prediction**, KDD 2025](https://eprints.lse.ac.uk/128236/) | Uses heterogeneous graph attention and cross-view alignment for IVS prediction; explicitly notes that different IVS regions are unevenly informed because of liquidity constraints. | This is the closest graph-attention anchor. LoG-IV should not claim heterogeneous graph attention for IVS as new. The distinction is treating liquidity marks as evidence about quote reliability under a leakage-controlled masked-reconstruction protocol. |
| [Yang et al., **HyperIV: Real-time Implied Volatility Smoothing**, ICML 2025](https://icml.cc/virtual/2025/poster/44077) | Uses a hypernetwork to construct complete arbitrage-free surfaces in real time from a small number of market observations. | HyperIV is a strong real-time and hard-constraint reference. LoG-IV is not primarily a latency paper; its claim is about irregular graph benchmarking, missingness, and liquidity-dependent noise. |
| [Zhuang and Wu, **Meta-Learning Neural Process for Implied Volatility Surfaces with SABR-induced Priors**, arXiv 2025](https://arxiv.org/abs/2509.11928) | Treats IVS reconstruction from sparse quotes as a meta-learning neural-process problem and uses SABR-induced priors. | This is an important sparse-quote completion baseline family, but it is not graph-specific and does not by itself define the leakage and liquidity-reliability protocol LoG-IV targets. |
| [Gatheral and Jacquier, **Arbitrage-free SVI volatility surfaces**, Quantitative Finance 2014](https://www.tandfonline.com/doi/abs/10.1080/14697688.2013.819986) | Provides a classical arbitrage-free SVI reference for volatility surface construction and calibration. | SVI/SSVI remain essential surface-fitting baselines and diagnostics; LoG-IV should not treat them as optional in final manuscript-level comparisons. |

These anchors imply that the intended contribution should be stated narrowly:

- a leakage-controlled masked-reconstruction benchmark for irregular option
  chains;
- a modeling view that treats liquidity marks as noisy evidence about quote
  reliability;
- experiments under realistic missingness, including liquidity-correlated and
  block-structured masks;
- diagnostics that compare reconstruction error with decoded-price and
  no-arbitrage behavior;
- out-of-distribution evaluation as evidence about robustness, not as a causal
  claim about cross-market risk transmission.

This boundary is important for positioning. The paper should not describe
"turning option chains into graphs," "using GNNs for IV surfaces," or "adding
no-arbitrage penalties" as the novelty. The novelty, if supported by
experiments, is the combination of leakage control, reliability-aware graph
modeling, and evaluation under market-relevant missingness.

The manuscript draft should cite the final published or official proceedings
versions where available. Preprints should be labeled as such, and claims about
data coverage, architectures, and guarantees should be checked against the paper
text before submission.

## Current Evidence Boundary

The expanded data gate is much stronger than the initial MVP:

- U.S. expanded silver data: 2,979,716 rows, 62 observation dates, 40
  underlyings, 2,480 surfaces, and 2,480 surfaces with at least 20 nodes.
- Japan expanded silver data: 531,280 rows, 31 observation dates, 249
  underlying or surface keys, 6,991 surfaces, and 6,179 surfaces with at least
  20 nodes.

This is enough for a first masked-reconstruction benchmark, but not enough
for market-cycle temporal generalization claims. The main limitation is still
time coverage: 62 U.S. observation dates are enough for Protocol A, not a
regime study.

The current A1 stratified run is preliminary but substantially stronger than
the earlier single-seed run:

- It uses 20 epochs and three seeds.
- Graph variants beat train-only kNN, within-surface kNN, and within-surface
  RBF on masked U.S. IV MAE under the stratified mask.
- Decoded regularization improves the best masked IV MAE and sampled diagnostic
  counts, but convexity violations remain high enough that strict
  no-arbitrage claims are not supported.

Do not use these results as final evidence without the liquidity-correlated and
block-wing mask regimes, raw SVI accounting, and stronger baselines.

## Headline Task

The first main task is **masked reconstruction**:

- The target is IV at observed option contracts.
- Masked input nodes remove IV, bid, ask, mid, spread, same-day decoded price,
  same-day quote-derived liquidity fields, and IV-derived Greeks from model
  inputs.
- Target values remain available only to the loss and metrics.
- Headline metrics are computed on masked nodes, not on all observed nodes.

This is the first task because it tests whether a representation uses neighboring
contracts, liquidity variables, and surface geometry without requiring a clean
next-day forecasting target.

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
must not be used as manuscript-level benchmark comparisons.

The current train-only kNN baseline is a useful lower bound, but it is not
sufficient for a LoG paper. SSVI, constrained SVI, and the heavier ML/graph
baseline family are P1 until `data_v1` passes:

- constrained SVI and SSVI calendar fitting;
- graph Laplacian smoothing;
- tabular LightGBM or XGBoost on geometry and liquidity features;
- DeepSets or Set Transformer as no-edge neural set baselines;
- vanilla GCN, GAT, and GraphSAGE variants as standard graph baselines.

A graph result that beats only global means is not sufficient. A publishable
result should beat the strongest relevant interpolation or surface-fitting
baseline on masked IV MAE or normalized decoded-price MAE.

## Model Hypotheses

- Liquidity-aware message passing should avoid over-propagating stale, wide, or
  low-volume quotes.
- Heteroscedastic modeling should improve low-liquidity and
  liquidity-correlated masks more than random masks.
- Graph operators should handle irregular strike and tenor grids more naturally
  than fixed-grid image models.
- Decoded-price regularization should reduce financially implausible surfaces
  without relying on embedding-distance proxies.

These hypotheses become paper claims only if they survive the benchmark gate.

The v2 model ladder is implemented as a registry rather than a default run
matrix. The registered variants separate no-liquidity, liquidity-feature-only,
current scalar liquidity gate, heteroscedastic-loss-only, reliability-gated
decoder, and full heteroscedastic configurations. The full research claim still
requires A1/A2 runs; the registry only makes the ablations auditable.

The repo also exposes related-work proxy variants through
`variant_suite=anchor_proxy`. These include proxies for Deep Smoothing,
Operator Deep Smoothing, Hexagon-Net, HyperIV, and sparse-quote neural-process
style completion, plus a native CNP baseline. The implemented backbones are:
fixed-grid CNN smoothing, continuous-coordinate kernel operator smoothing,
heterogeneous edge-family attention with a lightweight cross-view alignment
loss, attentive neural-process context modeling, and a HyperIV-positioning
proxy. They are useful for unified-protocol
comparison, but they must be described as faithful-spirit baselines rather than
paper reproductions unless the corresponding external implementation, training
objective, data assumptions, and evaluation protocol are matched and validated.

For HyperIV specifically, the repo provides a `hyperiv-compare` command that
records or runs an external-code adapter manifest. The native
`anchor_hyperiv_proxy` does not claim hard no-arbitrage guarantees or real-time
hypernetwork behavior.

## Metrics

Headline metrics:

- masked IV MAE;
- masked IV RMSE;
- model difference relative to train-only kNN;
- liquidity-stratified masked IV error.

Supporting diagnostics:

- normalized decoded-price error;
- calendar and strike-convexity violation counts on true-IV and predicted-IV
  decoded prices;
- vertical-spread diagnostics on synthetic and Europeanized subsets;
- error-versus-violation trade-off summaries;
- performance degradation under distribution shift, normalized against naive
  train-only baselines;
- out-of-distribution degradation from U.S. to Japan or held-out tickers;
- sensitivity to seed and training length.

Required real-data diagnostics:

- reliability ranking: Spearman or Kendall rank correlation between predicted
  precision and held-out absolute error on masked nodes;
- predicted-reliability buckets: masked IV MAE, normalized price MAE, and NLL by
  predicted precision bucket;
- liquidity buckets: masked IV MAE and heteroscedastic NLL by spread, volume,
  and open-interest bucket;
- interval diagnostics, if predictive intervals are emitted: empirical
  coverage and average interval width by liquidity bucket.

These diagnostics are mandatory for any claim that LAGOS-Hetero treats
liquidity as observation reliability. Lower MAE alone is not sufficient evidence
for the heteroscedastic interpretation.

Required graph-necessity ablations:

- no-edge set/context model;
- liquidity-as-feature-only GNN;
- scalar-gate GNN;
- random-edge or shuffled-edge GNN using the same node features;
- ODS-style continuous-coordinate operator without liquidity reliability.

These ablations are mandatory for any claim that graph structure contributes
beyond node features, set aggregation, or generic coordinate smoothing.
Synthetic prior pretraining is not required for the main real-data claim; it
should remain an appendix or future-work item unless it improves real-data
generalization without weakening the benchmark framing.

## Reviewer Risk Register

Likely LoG reviewer concerns:

- The paper may look like a finance application unless the graph learning
  benchmark and operator contribution are foregrounded.
- The data window may be considered too short for generalization claims.
- The baseline set may be considered weak without SVI/SSVI, interpolation,
  tabular ML, set models, and standard GNN baselines.
- The model comparison may be considered incomplete if it includes only simple
  means and kNN.
- Decoded-price diagnostics may be questioned unless forward, rate, dividend,
  and IV inversion assumptions are explicit.
- Japan out-of-distribution claims may be rejected if the market normalization
  and timestamp semantics are not documented.
- No-arbitrage claims may be rejected if convexity or calendar diagnostics are
  worse than strong baselines.
- The heteroscedastic contribution may be rejected if predicted precision does
  not rank realized errors or improve low-liquidity buckets.
- The graph contribution may be rejected if no-edge, shuffled-edge, or
  continuous-coordinate baselines perform similarly under the same masked
  reconstruction protocol.

Mitigations:

- Keep the main claim to leakage-controlled masked reconstruction of
  irregular option-surface graphs.
- Report fixed split manifests, train-only baseline fitting, and masked-node
  metrics as first-class artifacts.
- Run 100-epoch, 3-seed or larger experiments before final tables.
- Add the stronger baseline ladder before claiming graph superiority.
- Report liquidity, moneyness, tenor, option-type, ticker, and date-stratified
  metrics.
- Label Japan results as out-of-distribution evaluation until transfer evidence
  is strong.

## Intended Positioning

The current positioning is:

> We introduce LoG-IV, a leakage-controlled benchmark for irregular
> option-surface graphs with liquidity-dependent observation noise, and evaluate
> graph models against train-only interpolation, surface-fitting, tabular, set,
> and standard graph baselines on real U.S. options. Japan is used as a separate
> out-of-distribution evaluation setting.

This positioning is viable only if the final benchmark shows that the graph
model beats the strongest relevant baseline on masked U.S. test error and does
not materially worsen decoded-price diagnostics.

## Immediate Execution Plan

1. Keep the expanded U.S./Japan silver data gates green.
2. Add the stronger baseline ladder before paper-candidate runs.
3. Run `benchmark-protocol` with fixed temporal masked reconstruction.
4. Use capped runs only for pipeline checks.
5. Treat only multi-seed runs as candidates for manuscript-level evidence.
6. Diagnose GNN versus the strongest relevant baseline only after the protocol and data gates are already
   satisfied.
