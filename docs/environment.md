# Environment contract and decisions

The supplied architecture brief defines a sell-only task: sell parent quantity
`Q` BTC using marketable orders at 60 decisions, five seconds apart. The
implementation preserves this scope and provides the four proposed actions.

An older workbook, `~/AIDEN_From_Scratch_Implementation_Workbook.md` dated
13 September, specifies a different six-action synthetic buy-side simulator
(`HOLD`, `CANCEL`, `JOIN`, `IMPROVE`, `TAKE`, `SWEEP`). This implementation follows
the later pasted BTC replay brief. Supporting that workbook's passive actions
would require an explicit resting-order and fill model, plus different timing
and accounting assumptions. L2 replay cannot supply individual queue positions.

## What happens at a decision

1. Observe the recorded book known by decision time `t` and the trade aggregate
   received in `(t-5s, t]`.
2. Convert the action to a requested BTC quantity.
3. Walk bid levels from highest price to lowest, filling no more than the
   displayed BTC amount at each of the 25 levels.
4. Charge configured fees on executed USDT notional; update cash and inventory.
5. Advance one row, or terminate when inventory is gone or the last action ends.

The grid runs from `start` through `start + 295 seconds`. The final liquidation
attempt is at **295 seconds**, using that recorded book. There is no invented
snapshot or extra execution at 300 seconds. The time feature describes the
remaining scheduled slices: 300 seconds initially, 5 seconds before the last
action, zero in the terminal observation.

## Actions and their purpose

Let `x` be unsold BTC and `n = 60 - step_index` include the current decision.
The adaptive schedule is `s = x / n` BTC. Action IDs are stable integers.

| ID | Action | Requested BTC before the last step | Purpose |
| ---: | --- | --- | --- |
| 0 | `WAIT` | 0 | Defer execution while keeping inventory exposure |
| 1 | `BELOW_SCHEDULE` | `0.5 * s` | Trade more slowly than the current schedule |
| 2 | `ON_SCHEDULE` | `s` | Equal slices over the remaining opportunities |
| 3 | `ABOVE_SCHEDULE` | `min(x, 2 * s)` | Accelerate when the agent prefers the current book |

These multipliers are project choices, not documented RBC action parameters.
Each branch in `compile_action` explains its arithmetic, data structures, and
reason for inclusion. An `IntEnum` names the choices; scalar arithmetic sizes
orders; a frozen `ActionRequest` records the calculation; a linear scan prices
fills through the ordered book. No order queue is constructed.

For `x=6 BTC` with `n=60`, requests are `0`, `0.05`, `0.10`, and `0.20 BTC`.
If an on-schedule request cannot fill at all, the next schedule is `6/59 BTC`.
The catch-up rule spreads missed quantity over the remaining opportunities.

Every request is capped by remaining inventory. `ActionRequest.executable_btc`
also reports the current displayed-depth cap. The engine retains the larger
submitted quantity so `unfilled_btc` remains auditable; it never fills past level
25. Unfilled marketable requests expire after this tick. They do not become
resting limit orders or automatically execute on the next book.

On the last decision, every action requests all remaining inventory once. This
is a deadline rule, not a guarantee of completion. All four actions intentionally
become equivalent then; adding a fifth "liquidate" action would be redundant.
`force_terminal_liquidation=False` exists in the core for diagnostic tests; the
Gym adapter always uses the deadline rule.

"Sell immediately" is a baseline policy using `step(remaining_btc)`, not another
RL action. Buying, shorting beyond the parent, cancelling, and passive limit
orders are outside this brief's marketable-sell model.

## Accounting and execution

At level `i`, let `p_i` be USDT/BTC and `q_i` be displayed BTC. Sell
`f_i = min(remaining_request, q_i)` BTC and receive `f_i * p_i` USDT.

For bids `(100, 0.4), (99, 0.7), (98, 0.5)`, a `0.9 BTC` request receives
`0.4*100 + 0.5*99 = 89.5 USDT`. A `2 BTC` request fills only `1.6 BTC` for
`158.3 USDT`, leaving `0.4 BTC` unfilled.

All 25 bid and ask price/amount pairs must exist. Prices are positive and finite;
amounts are finite and nonnegative. Bids strictly decrease, asks strictly
increase, and the best bid is below the best ask. Zero displayed size is allowed
and contributes no fill. Missing or corrupt depth fails explicitly.

The engine stores immutable copies of input rows. `math.fsum` reduces cumulative
roundoff; a running request balance is recomputed from summed actual fills.
No tolerance converts unexecuted dust into sold BTC. Tests compare accounting
identities with numerical tolerance, without altering economic state.

For configured `fee_bps`, fee USDT equals `gross_usdt * fee_bps / 10_000`.
Fees are debited in USDT, so `net_usdt = gross_usdt - fee_usdt`. The default is
zero; this is an experimental assumption, not a verified historical Binance
fee. Every compared policy must use the same setting. Lot sizes, minimum order
notional, BNB fee discounts, and exchange acceptance rules are not modeled.

`StepResult` records strategy request, submitted quantity, actual fill, unfilled
request, inventory, gross/fee/net cash, action, and reward components. History is
one record per decision. No second execution happens on the final book.

## Reward: completion score versus economic marking

Let `P0` be the arrival midprice in USDT/BTC. A step filling `f` BTC and receiving
`c_net` USDT earns:

```text
execution_reward_bps = 10_000 * (c_net - f * P0) / (Q * P0)
terminal_penalty_bps = 10_000 * remaining_btc / Q  # only on termination
reward = execution_reward_bps - terminal_penalty_bps
```

The total **undiscounted** reward is:

```text
total_reward = 10_000 * (total_net_usdt / (Q * P0) - 1)
completion_score_bps = -total_reward
```

For completed orders, this is negative net sell implementation shortfall.
Higher proceeds for the same obligation produce higher reward. Use `gamma=1`
when implementing PPO if this undiscounted objective is intended: discounting
changes the relative importance of early costs and terminal loss.

For incomplete orders, the learning score gives **zero credit to unsold BTC**.
This is a deliberately severe completion convention, not an estimate that held
BTC has no economic value. A policy that sells nothing scores `10,000 bps` of
cost, even if the price rises. A 1% unsold remainder contributes `100 bps` of
terminal penalty; compare that magnitude with ordinary spread/fee costs. We
avoid a second inventory-risk or completion-penalty coefficient in this first
version. This prioritizes executable cash and can dominate small price gains;
it is a project objective, not a universal quant-finance reward standard.

The summary separately reports:

- Completion fraction and unfilled BTC.
- Gross proceeds, fees, net proceeds, and gross/net average execution prices.
- Gross and net arrival-price shortfall for **completed orders only**.
- Residual value marked at the last observed midprice, `x * P_last` USDT.
- Marked shortfall, `10_000 * (1 - (net_cash + residual_mark)/(Q*P0))`.
- Completion score and total reward, which are defined even when incomplete.

The midprice mark is an economic diagnostic, not an executable liquidation
price, cash receipt, or fee-bearing fill. Never pool only completed orders while
hiding incomplete ones, and never label the conservative incomplete score as
ordinary realized implementation shortfall.

## Ten causal observations

`features.py` fixes both the names and their order. Named values use these units;
`observation_vector` divides the four bps fields by 100 for the Gym array.

| Feature | Definition / purpose |
| --- | --- |
| Inventory fraction | `x/Q`; progress toward completion |
| Time fraction | `n/60`; remaining decision opportunities |
| Spread bps | `10_000*(ask-bid)/mid`; instantaneous spread cost |
| Depth imbalance | `(D_bid-D_ask)/(D_bid+D_ask)` across all 25 levels |
| Relative bid depth | `log(1+D_bid/Q)`; liquidity relative to the parent |
| Weighted-mid deviation bps | `(weighted_mid-mid)/mid * 10_000`; top-level pressure |
| Recent return bps | `10_000*log(mid_t/mid_previous)` |
| Trailing realized volatility bps | `10_000*sqrt(sum(log_return**2))`, at most 12 intervals |
| Recent trade volume | `log(1+volume_5s/Q)`; recent trading activity |
| Signed trade flow | `signed_volume_5s/volume_5s`; positive means buyer aggression |

`weighted_mid = (ask*bid_size + bid*ask_size)/(bid_size+ask_size)`.
It is a weighted-mid proxy, not an estimated Stoikov microprice. Its deviation
equals half the spread times top-level imbalance, so adding top-level imbalance
as an extra feature would repeat the same information. The implemented imbalance
uses full depth. Signed trade volume is not quote-event order-flow imbalance:
we do not reconstruct limit-order submissions and cancellations here.

Missing prior episode returns use zero initially; volatility warms up on only
available intervals within the episode and reaches a 60-second window after
12 intervals. Trade aggregates at the first decision still describe the previous
five seconds and are already known. Zero trade volume gives zero signed flow;
zero total depth gives zero imbalance; zero top size uses the ordinary midpoint.

Scaling is fixed arithmetic, with no statistics learned from any split. If a
fitted normalizer is introduced later, fit on training only and freeze it.
UTC timestamps and absolute BTC prices are absent from the Gym feature vector.
Terminal observations use the last executed row and time fraction zero, including
early completion; they do not reveal the next historical book.

## Replay assumptions and limits

- Orders execute instantaneously against the latest received book, at most five
  seconds old. Order-routing latency, hidden depth, and exchange rejection are absent.
- Every later snapshot is replayed unchanged by our fills, so liquidity can
  reappear on later ticks. This includes reusing a received snapshot if it is
  selected twice by preprocessing. This is an exogenous replenishment assumption.
- Walking displayed depth measures mechanical sweep cost; it does not estimate
  our own temporary or permanent market impact.
- Level 2 amounts aggregate price levels. No individual order queues,
  price-time-priority behavior, or passive execution probability is inferred.
- Quantity `Q` is supplied explicitly; the environment never sizes it using
  future episode volume. The `0.01 BTC` smoke-test size is not a research calibration.
- The ten observations are a compact partial view of market state. The model
  is not claimed to be a fully observed Markov market simulator.
- This is seven isolated days with one validation and one test day, not many
  independent months. No live-performance claim follows from replay results.

## Sources and what they justify

- [CFA Institute: Trading Costs and Electronic Markets](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/trading-costs-and-electronic-markets): distinguishing explicit fees, spread, impact, and unfilled-order costs.
- [CFA Research Foundation: Transaction Cost Measurement and Management](https://www.cfainstitute.org/sites/default/files/-/media/documents/book/rf-publication/2015/rf-v2015-n4-1-pdf.pdf): arrival-price benchmarks and separating execution from opportunity costs. The conservative zero-credit learning score is our design choice.
- [Gould and Bonart: Queue Imbalance as a One-Tick-Ahead Price Predictor](https://arxiv.org/abs/1512.03492): motivation for imbalance signals; not evidence our BTC feature set is profitable.
- [Cont, Kukanov, and Stoikov: The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402): quote-event OFI includes more information than the signed trade aggregate used here.
- [Stoikov: The Micro-Price](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694): distinction between a weighted midpoint and an estimated microprice.
- [Gymnasium API](https://gymnasium.farama.org/api/env/) and [finite horizon handling](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/): seeded reset, five-value step result, and termination at a task-defined deadline. The adapter returns `terminated=True, truncated=False` at completion or the 60-step deadline.

The original four-feature RBC notes mentioned in the brief were not present.
No claim is made that these ten features or four action multipliers reproduce them.
