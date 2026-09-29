# Supplementary VWAP benchmark

This comparison was added at the user's request **after** the original March
evaluation. It preserves the original PPO checkpoint, model selection, data,
order sizes, and fees. It does not turn the already inspected March day into a
new untouched test set.

## Measured results

The actual comparison covers **1,996 complete episodes** across all seven days,
three sizes, and three policies: **17,964 policy/episode/size records**. The
volume profile uses 85,440 training observations. Five midnight minutes had no
complete training windows and therefore retain the documented training-mean
fallback. No evaluated episode had zero market volume.

At the primary 0.7909 BTC size on March 1, with a 10-bp execution fee:

| Metric | Frozen PPO | VWAP schedule |
|---|---:|---:|
| Mean net arrival shortfall (bps) | 7.126 | 7.531 |
| P95 net arrival shortfall (bps) | 35.896 | 29.313 |
| Worst-5% mean arrival shortfall (bps) | 45.157 | 37.183 |
| Completed orders | 287 / 287 | 287 / 287 |
| Mean gross slippage versus market VWAP (bps) | 0.320 | 0.717 |
| Mean net slippage versus market VWAP (bps) | 10.319 | 10.716 |

PPO improves mean arrival cost by **0.405 bps**, winning **157/287 (54.7%)**
paired episodes, but its tail is worse. It also does not sell above the actual
market-VWAP price on average: its mean gross slippage is positive.

At 3.954 BTC, the VWAP schedule completes **287/287** orders while PPO completes
**280/287**. Their all-order conservative scores are **7.603** and **44.432 bps**,
respectively. The completed-only PPO cost excludes seven failures and should
not be used alone to claim large-order superiority.

The comparison is mixed across splits: at the primary size, February favors
the VWAP schedule (**9.375 bps**) over PPO (**9.516 bps**). On March, VWAP's mean
is close to TWAP's **7.504 bps**, although their schedules differ: the mean L1
distance between normalized VWAP weights and uniform TWAP weights is **0.192**.
These findings are descriptive, and no forecast setting or model was retuned
after seeing them.

## Two different comparisons

The [market VWAP definition](https://www.interactivebrokers.com/campus/glossary-terms/volume-weighted-average-price-vwap/)
is total traded notional divided by total traded quantity over a stated interval.
It is an ex-post price benchmark. An
[execution algorithm targeting VWAP](https://www.interactivebrokers.com/docs/general/order-types/algorithmic-orders/ib-algorithms/vwap)
must instead decide trades before it knows the interval's final volume or price.
This project reports both comparisons, without claiming to reproduce a broker's
VWAP algorithm or guarantee execution at the benchmark price.

1. **Executable historical-profile VWAP:** sell in proportion to a volume curve
   estimated only from the five training days. Compare its actual simulated
   fills against the frozen PPO and TWAP on identical episodes and obligations.
2. **Receipt-time market VWAP:** calculate the actual volume-weighted trade price
   over the episode's execution horizon, then measure each policy's execution
   price relative to it. This price never enters a policy's observations or sizing.

## Historical-volume schedule

For each UTC minute of the day, `fit_volume_profile` averages
`trade_volume_btc_5s` across the observed decision rows in complete training
episodes. A minute bucket pools up to 12 five-second observations per day. This
fixed one-minute smoothing is deliberately simple given only five training days;
it was not selected by comparing March outcomes.

The original aggregates describe `(t-5s, t]`. Their minute-of-decision averages
estimate local activity at each execution time. At arrival, the 60 weights are
looked up using the known decision clock, at `t0, t0+5s, …, t0+295s`. The schedule
does not inspect any future row of the evaluation episode. These are 60 local
activity weights, not a claim that 60 postarrival trade intervals are available.

At decision index `i`, the request is:

```text
requested_btc = remaining_btc * weight[i] / sum(weight[i:])
```

Recomputing from remaining inventory redistributes previous partial fills over
the remaining forecast curve. Equal weights reproduce adaptive TWAP exactly.
The formula needs a 60-element weight tuple and a short suffix sum; no order
queue, optimizer, or future volume estimate from the test episode is required.

Missing training observations are not zero volume: an unobserved minute uses
the global training mean, with support counts retained in `volume_profile.json`.
If all remaining weights are zero, use remaining-time TWAP. The shared execution
environment applies depth limits, fees, and the final liquidation attempt.

## Actual market VWAP and units

The price benchmark uses **rows 1 through 59**, covering `(t0, t0+295s]` by
local receipt time. Row zero's trades precede arrival and are excluded. There
is no invented trade bin or execution at 300 seconds. The same horizon is used
even when a policy finishes early.

```text
market_vwap = sum(trade_notional_usdt_5s) / sum(trade_volume_btc_5s)
sell_slippage_bps = 10_000 * (1 - average_execution_price / market_vwap)
```

The market price includes both trade sides. Do not average the precomputed
five-second VWAPs: intervals with different BTC volume need different weights.
No trades means an undefined benchmark, recorded as missing rather than zero.

Lower slippage is better for this sell task. Negative gross slippage means the
average sale price exceeded market VWAP. Net slippage uses fee-adjusted proceeds
and therefore includes the assumed 10-bp execution fee; the market transaction
price itself is a gross price. Both are reported for completed orders only.
Incomplete orders retain their completion status, residual BTC, and original
arrival-based conservative score in the all-episode comparison.

## Reproduce

From the repository root, using the full dependencies and original local data:

```bash
python benchmark_vwap.py --study artifacts --output runs/vwap_reproduction
```

The command requires a new output directory. It verifies the original artifact
bundle, selected checkpoint, frozen source, and all market-data hashes. It fits
and saves the training-only profile before computing comparisons, replays the
VWAP baseline on every complete episode at all three sizes, and reuses the
hash-verified original PPO/TWAP execution records. No PPO retraining or new seed
selection occurs.

Outputs include the forecast and protocol, per-episode results, exact paired
comparisons, actual first-episode VWAP fills, profile-versus-TWAP diagnostics,
summary tables, figures, and integrity hashes. The detailed measured supplement
is `artifacts/vwap/REPORT.md` in the optional local bundle. The main findings
also appear in [the README results](../README.md#15-what-the-measured-results-mean).

Only one held-out day is available. Small mean differences, weak historical
volume forecasts, incomplete orders, and tail risk all matter. Training-split
comparisons are in-sample diagnostics because their volume profile uses those
same training days. No statistical significance or live profitability is claimed.
