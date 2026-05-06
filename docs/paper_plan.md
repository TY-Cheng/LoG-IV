# Paper Plan

## Working Title

**LoG-IV: Leakage-Controlled Graph Learning for Irregular Option-Implied
Volatility Surfaces**

## Manuscript Thesis

LoG-IV should be written as a graph-learning benchmark and modeling paper. The
central object is an option chain represented as an irregular graph of option
tokens. The central empirical question is whether graph or token models can
reconstruct masked implied volatilities under leakage-controlled missingness
more accurately than interpolation and surface-fitting baselines, especially
when quote reliability varies with liquidity.

This is not a trading, alpha, hedging, or live risk-management paper. There is
no backtest in the first manuscript. Japan is an out-of-distribution evaluation
setting, not the basis for a cross-market causal claim.

## 1. Introduction

### 1.1 Motivation

Real option chains are sparse and irregular. Strike grids differ by expiry,
maturity grids are uneven, and quote quality varies with bid-ask spread, volume,
open interest, and market conventions. Standard image-grid surface models hide
some of this structure by interpolating onto fixed grids before learning. LoG-IV
instead treats the option chain as an irregular graph or token set.

The paper should foreground three practical constraints:

- option-token geometry is irregular in strike and tenor;
- missingness is structured, not purely random;
- observed quotes are noisy, and liquidity variables are plausible indicators
  of observation reliability.

### 1.2 Core Research Question

Can graph models reconstruct sparse and irregular option-implied-volatility
surfaces under leakage-controlled masking and liquidity-dependent observation
noise more accurately than train-only and within-surface interpolation
baselines?

The question has three separable parts:

- **Representation:** does the graph or token representation help beyond
  fixed-grid or no-edge set baselines?
- **Reliability:** do liquidity marks help as reliability signals, not only as
  ordinary node features?
- **Protocol:** do the gains survive realistic missingness and strict leakage
  controls?

### 1.3 Contributions

The contribution claim should be narrow and auditable:

1. A leakage-controlled masked-reconstruction benchmark for irregular
   option-surface graphs.
2. A graph construction and masking contract that prevents masked query nodes
   from carrying same-day quote-derived target information.
3. A model and ablation ladder for liquidity-aware graph learning, including
   liquidity-feature-only, scalar-gate, heteroscedastic-loss, reliability-gated,
   random-edge, shuffled-edge, set-context, and decoded-regularized variants.
4. Evaluation under market-relevant missingness regimes:
   `stratified`, `liquidity_correlated`, and `block_wing`.
5. Diagnostics that report reconstruction error together with normalized
   decoded-price error, reliability diagnostics, and sampled no-arbitrage
   violation counts.

### 1.4 Claim Boundary

Do not claim novelty for any of the following by themselves:

- using neural models for implied-volatility surfaces;
- turning option chains into graphs;
- adding no-arbitrage penalties;
- using liquidity variables as extra features.

The defensible novelty, if the evidence gate passes, is the combination of
leakage-controlled masking, irregular graph evaluation, liquidity-reliability
ablation, and missingness-aware diagnostics.

## 2. Literature Review

### 2.1 Neural Implied-Volatility Smoothing

[Ackerer, Tagasovska, and Vatter, **Deep Smoothing of the Implied Volatility
Surface**, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/858e47701162578e5e627cd93ab0938a-Abstract.html)
is the core neural-smoothing anchor. It establishes that neural models can fit
and smooth implied-volatility surfaces with soft financial constraints. LoG-IV
should not present neural IV smoothing as new.

[Yang et al., **HyperIV: Real-time Implied Volatility Smoothing**, ICML
2025](https://icml.cc/virtual/2025/poster/44077) is a strong real-time and
hard-constraint anchor. LoG-IV is not primarily a latency paper and should not
claim hard no-arbitrage guarantees unless the corresponding assumptions and
checks are implemented.

### 2.2 Graph and Operator Models for Irregular Option Data

[Wiedemann, Jacquier, and Gonon, **Operator Deep Smoothing for Implied
Volatility**, ICLR
2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f115f619b62833aadc5acb058975b0e6-Abstract-Conference.html)
is the closest graph/operator method anchor. It maps irregular observed option
data to smoothed surfaces. LoG-IV should differentiate through leakage control,
missingness regimes, and liquidity-reliability modeling rather than through the
existence of an irregular operator model.

[Liang et al., **Hexagon-Net: Heterogeneous Cross-View Aligned Graph Attention
Networks for Implied Volatility Surface Prediction**, KDD
2025](https://eprints.lse.ac.uk/128236/) is the closest heterogeneous graph
attention anchor. It explicitly motivates uneven information across IV surface
regions because of liquidity constraints. LoG-IV should treat this as prior art
for graph attention and focus its own claim on benchmark protocol and
reliability ablation.

### 2.3 Sparse-Quote Completion and Neural Processes

[Zhuang and Wu, **Meta-Learning Neural Process for Implied Volatility Surfaces
with SABR-induced Priors**, arXiv 2025](https://arxiv.org/abs/2509.11928)
anchors the sparse-quote neural-process family. LoG-IV should include CNP/ANP
style baselines or proxies, but should describe them as sparse completion
comparators rather than graph methods.

### 2.4 Classical Surface Fitting and No-Arbitrage References

[Gatheral and Jacquier, **Arbitrage-free SVI volatility surfaces**,
Quantitative Finance 2014](https://www.tandfonline.com/doi/abs/10.1080/14697688.2013.819986)
is the classical no-arbitrage SVI anchor. SVI/SSVI accounting remains necessary
for manuscript-level comparisons. Raw SVI failure rates, timeouts,
underidentified slices, and constrained/projection variants should be reported
explicitly rather than hidden behind silent fallback behavior.

### 2.5 Positioning Relative to Prior Work

LoG-IV should be positioned as a benchmark and evaluation contribution:

- prior work establishes neural and graph methods for IV smoothing;
- LoG-IV asks whether those families remain strong under leakage-controlled
  masked reconstruction and market-relevant missingness;
- liquidity is evaluated as noisy evidence about quote precision, not asserted
  as a causal mechanism.

## 3. Materials and Methods

### 3.1 Data

The main empirical material is the expanded U.S. option silver table:

```text
data/silver/option_quotes/us_option_quotes_expanded.parquet
```

Current `data_v1` status:

- 2,979,716 deduplicated U.S. option rows;
- 40 underlyings;
- 62 usable observation dates from 2026-02-02 through 2026-04-30;
- 2,480 usable `(underlying, observation_date)` surfaces after
  `min_nodes_per_surface=20`;
- IV source: `option_mid_price_with_underlying_daily_close`;
- IV method: Black-forward bisection with zero rate and zero dividend.

The out-of-distribution material is the expanded Japan option silver table:

```text
data/silver/option_quotes/jp_option_quotes_expanded.parquet
```

Current Japan status:

- 531,280 deduplicated rows;
- 31 usable option observation dates;
- 6,179 usable surfaces after the same minimum-node gate.

Synthetic-LoG-IV is a reproducibility and controlled-diagnostic material. It
should be used for oracle diagnostics and no-arbitrage sanity checks, not as a
substitute for real U.S. benchmark evidence.

### 3.2 Preprocessing

The canonical option token is:

```text
(market, underlying, observation_date, expiry, strike, option_type)
```

Preprocessing steps to report:

1. Ingest vendor option rows into bronze manifests.
2. Normalize rows into canonical silver option tokens.
3. Deduplicate by canonical token key.
4. Join same-date underlying daily close as the current spot/forward proxy.
5. Construct mid prices and relative spreads where bid/ask are available.
6. Infer IV by Black-forward bisection when vendor IV is unavailable.
7. Compute tenor, log moneyness, quote-liquidity fields, and surface IDs.
8. Filter to usable surfaces with at least `min_nodes_per_surface=20`.

The current IV inversion uses a zero-rate, zero-dividend proxy. This is
acceptable for engineering targets but not enough for strict no-arbitrage or
pricing claims. The manuscript must state this limitation clearly.

### 3.3 Feature Engineering

Each node represents one option token. Candidate model features are grouped as:

- **geometry:** log moneyness, tenor, strike/expiry-derived coordinates;
- **contract metadata:** option type and market/underlying/date identifiers
  used for grouping or embedding where allowed;
- **visible quote features:** IV, bid, ask, mid, spread, volume, open interest,
  and liquidity score on visible context nodes;
- **target fields:** IV and decoded normalized price, available only to loss,
  metrics, and diagnostics for masked nodes.

Masked query nodes must not carry same-day quote-derived fields:

- IV;
- bid, ask, mid, spread;
- volume, open interest;
- decoded price;
- quote-derived liquidity score;
- IV-derived Greeks.

Visible-context aggregates must be computed after masking and only from visible
nodes. Normalization must use train-only statistics or visible-only statistics
under the current mask.

### 3.4 Graph Construction

The graph is built at the option-surface level. A surface is a
`(market, underlying, observation_date)` collection of option tokens.

Current implemented edge families:

- `strike_neighbor`: adjacent strikes within the same expiry and option type;
- `maturity_neighbor`: adjacent expiries for comparable strike and option type;
- `liquidity_similarity`: nearby nodes in liquidity and geometry profile.

Planned or diagnostic edge families include delta-neighbor and calendar-bucket
edges, but these require stronger Greeks and forward-curve assumptions before
being promoted to main evidence.

### 3.5 Main Task

The main task is **masked reconstruction**:

1. Deterministically select masked target nodes per surface and seed.
2. Remove disallowed same-day quote-derived fields from masked query inputs.
3. Predict IV for the masked nodes.
4. Compute headline metrics only on masked nodes.

Supported mask regimes:

- `stratified`: bucketed by liquidity, moneyness, tenor, and option type;
- `liquidity_correlated`: preferentially masks low-volume and low-open-interest
  nodes;
- `block_wing`: masks largest absolute log-moneyness wing nodes;
- `random`: engineering diagnostic only.

`stratified`, `liquidity_correlated`, and `block_wing` form the Protocol A
manuscript mask set. `random` should not be a main claim.

### 3.6 Splits, Resampling, and Statistical Design

Primary split:

- `temporal`: train, validation, and test are separated by observation date.

Robustness splits:

- `ticker_holdout`: selected underlyings are withheld from training;
- `temporal_ticker_holdout`: temporal split plus held-out ticker evaluation.

Engineering-only split:

- `random`: useful for debugging, not for manuscript claims.

Every run writes `splits.json` with train, validation, test, masked count, and
held-out ticker metadata. The current A1 temporal split uses 1,680 training
surfaces, 400 validation surfaces, and 400 test surfaces.

Statistical design:

- pipeline checks may use one seed and one or two epochs;
- manuscript-candidate evidence should use at least three seeds;
- aggregate tables should report mean and standard deviation across seeds;
- if confidence intervals are added, use paired bootstrap over evaluation
  surfaces or surface-date blocks, not row-wise bootstrap over option tokens;
- bootstrap is not currently part of the recorded benchmark artifacts and must
  be labeled as an added analysis if introduced.

### 3.7 Models and Baselines

Neural model families:

- `encoder_mlp`: no graph, token encoder baseline;
- `set_context_mlp`: no-edge set/context baseline;
- `gnn_no_liq`: graph model without liquidity features;
- `gnn_liq`: graph model with liquidity features;
- `gnn_decoded_calendar_convexity`: graph model with decoded calendar and
  convexity regularization;
- `lagos_liq_feature_only`: liquidity features without reliability gate;
- `lagos_scalar_gate`: scalar liquidity gate;
- `lagos_loss_only`: heteroscedastic loss without reliability gate;
- `lagos_attn_only`: reliability-gated attention without heteroscedastic loss;
- `lagos_hetero_full`: heteroscedastic loss plus reliability gate;
- `lagos_random_edges` and `lagos_shuffled_edges`: graph-topology negative
  controls.

Train-only baselines:

- global train mean IV;
- train mean IV by underlying;
- train mean IV by moneyness-tenor bucket;
- train-fitted moneyness-tenor kNN;
- random uniform IV.

Within-surface baselines using only visible evaluation nodes after masking:

- within-surface kNN;
- within-surface RBF interpolation;
- within-surface local linear interpolation;
- raw SVI per-expiry with timeout and failure accounting;
- constrained and calendar-projected SVI/SSVI variants when promoted.

Related-work proxy baselines:

- fixed-grid CNN smoothing proxy for Deep Smoothing / HyperIV-style families;
- continuous-coordinate operator proxy for Operator Deep Smoothing;
- heterogeneous edge-family attention proxy for Hexagon-Net;
- CNP/ANP-style neural-process baselines for sparse quote completion.

These proxy variants must be labeled as faithful-spirit comparators unless the
external code, data assumptions, objective, and evaluation protocol are matched.

### 3.8 Loss Functions and Training

The default training objective is a weighted sum of implemented terms:

- masked IV reconstruction loss;
- optional geometry and liquidity reconstruction losses;
- optional heteroscedastic negative-log-likelihood-style precision loss;
- optional decoded Black-forward calendar and strike-convexity regularizers;
- optional graph smoothness;
- optional cross-view alignment for the Hexagon-style proxy;
- optional contrastive loss across surfaces.

Important reporting rules:

- validation losses are not always comparable across model families when
  heteroscedastic or auxiliary terms differ;
- headline comparisons should use post-run masked IV metrics and diagnostics,
  not raw training loss alone;
- decoded no-arbitrage regularizers are soft training aids, not hard
  arbitrage-free guarantees;
- put-call parity remains diagnostic until rates, dividends, forwards, and
  option-style assumptions are explicit.

Training controls:

- deterministic mask and split construction under configured seeds;
- NumPy and PyTorch seeding before model construction;
- fixed epoch budgets for clean comparison unless early stopping is explicitly
  registered and applied consistently;
- default A1 candidate budget: 20 epochs; longer candidate budget: 50 or more
  only if validation curves justify it.

### 3.9 Evaluation Metrics

Headline metrics:

- masked IV MAE;
- masked IV RMSE;
- masked p90 absolute error;
- difference versus train-only moneyness-tenor kNN;
- difference versus strongest relevant within-surface baseline.

Supporting metrics:

- normalized decoded-price MAE/RMSE;
- by-split, by-underlying, by-option-type, by-moneyness, by-tenor, and
  by-liquidity bucket errors;
- OOD degradation ratios normalized against train-only baselines.

Reliability diagnostics:

- predicted precision versus realized masked absolute error rank correlation;
- masked IV MAE and normalized price MAE by predicted precision bucket;
- heteroscedastic NLL by spread, volume, and open-interest buckets;
- interval coverage and interval width if predictive intervals are emitted.

No-arbitrage diagnostics:

- decoded calendar monotonicity violation counts and severity;
- butterfly / strike-convexity violation counts and severity;
- vertical-spread diagnostic counts;
- put-call residuals when the assumptions are explicit enough.

These diagnostics are post-run evidence. They should be reported alongside
headline error, not merged into a single score unless a pre-registered scorecard
is defined.

### 3.10 Backtest Setup

No trading backtest is part of the first paper.

The manuscript should explicitly state that:

- the evaluation is masked reconstruction and robustness/OOD evaluation;
- there is no portfolio construction, signal generation, hedging simulation, or
  transaction-cost model;
- lower reconstruction error is not evidence of trading alpha;
- backtesting would require a separate timestamp, execution, liquidity, and
  risk-control protocol.

## 4. Experimental Design

### 4.1 Evidence Gates

| Gate | Claim status | Requirement |
| --- | --- | --- |
| Engineering check | Code path works | Tests, docs build, source probes, and small artifacts pass. |
| Data gate | Benchmark may run | U.S. has at least 2,400 usable surfaces and Japan has at least 20 usable dates. |
| Protocol gate | Results are interpretable | Fixed splits, masked reconstruction, train-only baselines, leakage diagnostics, and artifact manifests are recorded. |
| Benchmark gate | Manuscript-level evidence | Multi-seed runs beat kNN/interpolation baselines on masked-node metrics without materially worse diagnostics. |
| Claim gate | Manuscript claim | Liquidity, graph-necessity, no-arbitrage, and distribution-shift diagnostics support the interpretation. |

No result should be described as manuscript-level evidence before the benchmark
gate.

### 4.2 Main Experiment Matrix

| Experiment | Purpose | Current status | Promotion criterion |
| --- | --- | --- | --- |
| A1 stratified, 20 epochs, 3 seeds | Establish first multi-seed U.S. masked-reconstruction benchmark | Complete preliminary evidence | Graph variants beat train-only and within-surface baselines. |
| 2-epoch broad model screen | Check implemented model families and ablations | Complete engineering screen | Select candidates only; not a claim. |
| 2-epoch top4 under `liquidity_correlated` and `block_wing` | Test robustness to harder missingness | Complete engineering screen | Candidate must beat train-only and within-surface baselines on harder masks. |
| Top2 candidate runs under harder masks | Longer confirmation for `lagos_liq_feature_only` and `gnn_decoded_calendar_convexity` | Pending | Robust error improvement plus acceptable diagnostics. |
| Raw SVI accounting | Classical calibration baseline with failure rates | Pending | Report failures and successes directly; no silent fallback. |
| Ticker-holdout / temporal-ticker-holdout | OOD and cross-underlying robustness | Pending | Degradation should be normalized against simple baselines. |
| Related-work proxy suite | Unified-protocol method orientation | Partially pending | Label as proxy unless exact reproduction is validated. |

### 4.3 Current Candidate Decision

Current two-epoch evidence supports the following promotion decision:

- promote `lagos_liq_feature_only` as the strongest short-run masked-IV
  candidate across `stratified`, `liquidity_correlated`, and `block_wing`;
- promote `gnn_decoded_calendar_convexity` as the best diagnostics candidate,
  especially for normalized price error and sampled no-arbitrage violations;
- do not promote `lagos_loss_only` yet, because it wins the stratified short
  run but fails key baselines under `liquidity_correlated` and train kNN under
  `block_wing`;
- keep `gnn_liq` as a reference, not as a primary promoted candidate.

The current decision is still preliminary because the harder-mask top4 screen is
single-seed and two epochs.

### 4.4 Manuscript Tables and Figures

Main tables:

1. Dataset summary and data gates.
2. Model and baseline taxonomy.
3. Main masked-IV benchmark under `stratified`, `liquidity_correlated`, and
   `block_wing`.
4. Baseline ladder including raw SVI failure accounting.
5. Graph-necessity and liquidity-reliability ablations.

Main figures:

1. Option surface as an irregular graph.
2. Leakage-controlled masking diagram.
3. Error by liquidity and moneyness/tenor bucket.
4. Error versus decoded no-arbitrage violation trade-off.
5. Reliability calibration plot if precision diagnostics pass.

Appendix tables:

- per-seed metrics;
- per-underlying and per-date breakdowns;
- no-arbitrage diagnostics on true-IV versus predicted-IV decoded prices;
- OOD and ticker-holdout degradation ratios;
- hyperparameter and artifact manifest summary.

## 5. Results Reporting Plan

Results should be written in this order:

1. Data gate and split integrity.
2. Main masked IV reconstruction performance.
3. Harder missingness regimes.
4. Baseline and SVI comparison.
5. Liquidity and reliability diagnostics.
6. Graph-necessity ablations.
7. Decoded-price and no-arbitrage diagnostics.
8. OOD and robustness checks.

Each result paragraph should answer:

- what was compared;
- which split, mask, seed count, and epoch budget were used;
- which baseline is the relevant comparator;
- whether the result supports a benchmark claim, a modeling claim, or only an
  engineering observation.

## 6. Discussion Plan

The discussion should separate interpretation from evidence:

- If graph variants beat baselines across masks, discuss irregular graph
  representation as useful for sparse option-surface reconstruction.
- If liquidity-feature-only beats stronger reliability mechanisms, state that
  liquidity is useful as an input but heteroscedastic reliability modeling has
  not yet been established.
- If decoded regularization improves diagnostics but not headline MAE, present
  it as a trade-off rather than a pure win.
- If no-arbitrage violations remain frequent, avoid static-arbitrage claims and
  frame diagnostics as surface-quality checks.
- If Japan or ticker-holdout results are weak, present them as robustness limits
  rather than failed transfer claims.

## 7. Reviewer Risk Register

Likely reviewer concerns:

- The paper may look like a finance application unless the graph benchmark is
  foregrounded.
- The data window may be too short for market-cycle generalization.
- Baselines may be considered weak without SVI/SSVI, interpolation, tabular,
  set, and standard GNN comparisons.
- Heteroscedastic claims may be rejected if predicted precision does not rank
  realized errors.
- Graph claims may be rejected if no-edge, random-edge, shuffled-edge, or
  coordinate-operator baselines perform similarly.
- Decoded-price and no-arbitrage diagnostics may be questioned without explicit
  rate, dividend, forward, and option-style assumptions.
- Japan OOD claims may be rejected if timestamp semantics and market
  normalization are not documented.

Mitigations:

- Keep the main claim to leakage-controlled masked reconstruction of irregular
  option-surface graphs.
- Report split manifests, mask construction, train-only baseline fitting, and
  masked-node metrics as first-class artifacts.
- Run at least three seeds for manuscript-candidate tables.
- Add raw SVI accounting before final claims.
- Report liquidity, moneyness, tenor, option-type, ticker, and date-stratified
  metrics.
- Label related-work proxy variants as proxies unless exact reproduction is
  validated.

## 8. Immediate Execution Plan

1. Run longer candidate top2 masks:

   ```bash
   just benchmark-a1 mask=liquidity_correlated seeds=1 epochs=20 variant_suite=anchor_proxy variants=lagos_liq_feature_only,gnn_decoded_calendar_convexity out=reports/runs/candidate_top2_liquidity_correlated_e20 baseline_preset=fast no_arb_surfaces=50 skip_ood=true
   just benchmark-a1 mask=block_wing seeds=1 epochs=20 variant_suite=anchor_proxy variants=lagos_liq_feature_only,gnn_decoded_calendar_convexity out=reports/runs/candidate_top2_block_wing_e20 baseline_preset=fast no_arb_surfaces=50 skip_ood=true
   ```

2. Run raw SVI accounting on the promoted setting:

   ```bash
   just benchmark-a1 mask=stratified baseline_preset=full out=reports/runs/benchmark_a1_full
   ```

3. If the top2 candidates remain stable, rerun selected masks with at least
   three seeds.
4. Add ticker-holdout or temporal-ticker-holdout after the main U.S. Protocol A
   tables are stable.
5. Update `docs/results_snapshot.md` after each completed benchmark family;
   keep this paper plan focused on manuscript design and evidence gates.

## 9. Venue Positioning

LoG is appropriate only if the manuscript is framed as graph/geometric ML:
irregular option-token graphs, leakage-controlled graph benchmarks,
missingness-aware evaluation, and reliability-aware message passing. If the
final evidence is mixed, use a non-archival or extended abstract framing rather
than overstating a full benchmark claim.

Before submission, re-check the current LoG call for papers and formatting
requirements. The paper plan should not rely on stale venue details.
