# Reproducible experiment protocol

This protocol is fixed before opening the March test day for policy evaluation.
The environment's accounting and replay assumptions are in [environment.md](environment.md).

## Inputs and calibration

Use the actual local `data/*/*_5s.csv.gz` files, with entire-day splits:
September through January for training, February for selection, March for the
final test. These are seven isolated days. Hashes of the five training inputs
are in `configs/experiment.json`; evaluation records its own input manifest.

Calibration uses only the 85,440 decisions in 1,424 complete training episodes.
Median displayed 25-level bid depth is 7.9085725 BTC. Round 1%, 10%, and 50% of
that median to four significant digits: **0.07909, 0.7909, and 3.954 BTC**.
Each episode receives the same fixed quantity for every compared policy. No
future trade volume determines the parent size.

The primary experiment uses the middle size and an assumed constant **10 bps**
taker fee. This is a declared cost scenario, not measured historical Binance
account pricing. The final results also show 0/10/20-bps fee sensitivity. Fees
do not change observations or execution quantity, so fixed-policy cash can be
repriced from recorded gross proceeds without rerunning or retuning the policy.

## Policies and comparison fairness

- **Immediate:** request all remaining BTC now; retry a partial residual at the
  next recorded decision until completed or at the common deadline.
- **TWAP:** request remaining BTC divided by remaining decisions.
- **Liquidity heuristic:** multiply the TWAP request by current bid depth /
  arrival bid depth, bounded to [0.5, 2]. Use multiplier 1 if arrival depth is
  zero. This is a fixed, unoptimized rule.
- **PPO:** choose one of WAIT, BELOW_SCHEDULE, ON_SCHEDULE, ABOVE_SCHEDULE.
- **Wait control:** request nothing until the common terminal liquidation.
- **Above-schedule control:** always request twice remaining-time TWAP.

The last two controls were added after observing seed 7's validation behavior,
before any March policy evaluation. They distinguish a learned timing signal
from simply delaying or accelerating the whole order. They do not change the
training objective, hyperparameters, checkpoint selection, or test order sizes.

Every policy uses the same book sweep, arrival benchmark, parent quantity, fee,
and final liquidation rule. Immediate is an unrestricted aggressive execution
reference: it can sell the entire parent at the first decision, while the four
PPO actions limit normal urgency to twice adaptive TWAP. Report that structural
advantage explicitly when comparing them. It is useful to learn whether the
timing problem adds value relative to simply selling now.

## Actor-critic and PPO

The learned policy uses Stable-Baselines3's tested PPO implementation. The
project implements the data, execution environment, policies, training wiring,
selection, and analysis; it does not claim to implement PPO's optimizer from
scratch. The [PPO paper](https://arxiv.org/abs/1707.06347) defines the clipped
policy objective; [the library documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
describes its implementation and supported observation/action spaces.

The actor and critic each have two 64-unit Tanh layers. The actor outputs four
action probabilities; training samples from them, while evaluation takes the
most probable action. The critic predicts expected remaining cumulative reward
from the observation. It does not predict tomorrow's BTC price.

For action probability ratio `r = pi_new(a|s) / pi_old(a|s)` and estimated
advantage `A`, PPO maximizes the smaller of `r*A` and
`clip(r, 1-epsilon, 1+epsilon)*A`. The clipping discourages large policy changes.
The critic is fitted to return targets; an entropy term discourages premature
collapse to one action. Generalized advantage estimation smooths the noisy
advantage targets using the configured lambda.

Initial fixed settings: learning rate 0.0003, 1,024 rollout steps, minibatches
of 256, 10 optimization epochs per rollout, clip 0.2, entropy coefficient 0.01,
value-loss coefficient 0.5, gradient-norm cap 0.5, and GAE lambda 0.95. Gamma is
**1.0**, matching the undiscounted execution objective. Observations retain the
fixed feature scales already defined in `features.py`; there is no fitted test
normalizer. CPU runs use controlled Torch thread counts and seeds 7, 19, 42.

Train each seed for at least 200,000 interactions (the rollout boundary can
round this upward). Save both the final checkpoint and the best trained
checkpoint by deterministic validation score. Before any updates, log an
untrained reference, but do not select it as a trained checkpoint.

## Selection and held-out evaluation

During training, score a fixed set of 32 evenly spaced February episodes at
approximately 50,000-step intervals. Record the exact indices and actual
timesteps. For each seed, keep the lowest mean conservative completion score.
Next evaluate those three checkpoints on all 285 February episodes and choose
the lowest mean score at the primary order size. Ties use the lower seed.

Freeze the winning model's hash, all checkpoint hashes, config, and validation
selection results before final testing. The final March evaluation uses all
287 complete episodes and all three order sizes. Other seed results are
reported as sensitivity; the best March seed must not replace the model
selected on February. Training-day evaluations diagnose in-sample behavior.

Save per-episode raw results and the first episode's per-step trace. Report
completion alongside cost, plus mean/median/tail execution shortfall, worst-5%
cost, paired differences against TWAP, action usage, and execution timing.
Show both complete-only net shortfall and the all-episode conservative score;
never silently discard failed orders.

Group by fixed training-calibrated spread and volatility thresholds. Full
episode volatility is an **after-the-fact descriptor**, not a policy input.
Report its formula and do not imply it was observable when an order began.

Use descriptive distributions, not an i.i.d. significance claim from hundreds
of correlated five-minute windows. A single held-out market day cannot support
robust market-regime generalization. A negative or mixed PPO result remains a
valid project result, especially in replay without endogenous market impact.
