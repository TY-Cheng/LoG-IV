# Future Work

This page records extensions that should stay outside the first paper unless
they become necessary for interpretation. The current paper should remain
focused on masked reconstruction for irregular option surfaces.

| Extension | Research question | Gate |
| --- | --- | --- |
| Rate/dividend/forward upgrade | Do better curve and dividend assumptions materially change inferred IVs and decoded-price diagnostics? | After the masked reconstruction benchmark is stable. |
| Vendor-IV or richer quote source | Do vendor IV, quotes, Greeks, or open-interest histories change the benchmark ordering? | After source coverage and timestamp semantics pass the same expanded-silver gate. |
| Intraday option graphs | Do graph operators handle asynchronous quote updates better than EOD models? | After EOD schemas, masks, and baselines are stable. |
| Strong static-arbitrage regularization | Does explicit no-arbitrage regularization improve out-of-sample surfaces? | After decoded-price diagnostics are registered and rate/dividend assumptions are explicit. |
| Japan option transfer | Can U.S.-trained representations adapt to Japanese option chains? | After Japan has matched baselines, fixed out-of-distribution splits, and market-specific normalization assumptions. |
| Downstream Japan risk task | Can U.S. option representations improve Japan realized-volatility or tail-risk targets? | If direct Japan option transfer remains too fragile. |
| Submission package | Can every paper figure and table be regenerated from documented commands and source manifests? | Before manuscript circulation. |

Do not use these extensions to dilute the first paper. Trading PnL, live risk
monitoring, and execution-cost analysis are separate projects.
