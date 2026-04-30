# Graph Design

This page defines the first graph contract. It is deliberately model-agnostic so
the same data can feed smoothing baselines, token encoders, and GNNs.

## Nodes

Each node is one option token:

```text
v_j = (market, underlying, observation_date, expiry, strike, option_type)
```

Core node features:

- log moneyness, preferably `log(K / F)`;
- tenor in years;
- call-put indicator;
- implied volatility or normalized price target;
- bid-ask spread or relative spread;
- volume;
- open interest;
- missingness and stale-quote masks;
- masked-target visibility flag for masked reconstruction tasks.

Model inputs should preserve liquidity fields instead of using them only for
filtering. Low-liquidity quotes may be noisy but are part of the graph-domain
shift problem.

## Edge Families

Initial edge types:

| Edge type | Definition | Purpose |
| --- | --- | --- |
| `strike_neighbor` | Adjacent strikes within the same underlying, date, expiry, and option type | Local smile geometry. |
| `maturity_neighbor` | Adjacent expiries for the same underlying, date, strike, and option type | Calendar structure. |
| `delta_neighbor` | Nearby delta buckets where Greeks are available or computed | Cross-strike comparability. |
| `calendar_neighbor` | Same moneyness bucket across maturities | Term-structure smoothing. |
| `liquidity_similarity` | Similar liquidity and geometry profile | Noise-aware information sharing. |

The current code implements `strike_neighbor`, `maturity_neighbor`, and
`liquidity_similarity` edges. Node IDs preserve input order, so row `i` in the
feature tensor maps to `graph.nodes[i]`. Strike and maturity edges are
bidirectional. Delta and calendar-bucket edges remain planned until Greeks and
forward curves are defined.

## Edge Weights

Edge weights should be explicit. Candidate components:

- distance in log moneyness;
- distance in tenor;
- bid-ask spread penalty;
- volume and open-interest support;
- stale quote penalty.

Do not silently drop edge weights after construction. The current PyG operator
attaches edge weights to edge stores, consumes them through an incoming-edge
gate, and uses them in the smoothness regularizer. Run manifests record this as
an engineering implementation detail, not as paper evidence.

## Masking Semantics

For masked reconstruction, target nodes keep their option-token geometry and
liquidity context, but IV, bid, and ask are removed from model inputs. Target
quotes remain available only to the loss, metrics, and diagnostics. This rule is
part of the benchmark protocol and should not be relaxed in graph construction.

## No-Arbitrage Hooks

No-arbitrage checks are diagnostics at initialization. Candidate checks:

- call price nonincreasing in strike;
- put price nondecreasing in strike;
- convexity across strikes;
- calendar monotonicity for comparable moneyness;
- put-call parity residuals when rates, dividends, and forwards are available.

For benchmark protocol runs, training uses only decoded Black-forward calendar
total-variance and strike-convexity penalties where enough same-surface nodes
exist. Embedding-distance put-call parity and embedding-norm convexity proxies
are not paper-facing regularizers. Put-call parity remains a post-run diagnostic
until rate, dividend, and forward assumptions are explicit.

Paper-facing no-arbitrage diagnostics must be reported separately from training
regularizer loss and must compare true-IV and predicted-IV decoded prices.
