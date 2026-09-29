# Environment verification record

This document preserves the **historical 2026-09-21 environment audit**. Its
test count and smoke-check results describe that version of the project, not
the current expanded suite. The completed research study is recorded below
and in [results](../README.md#15-what-the-measured-results-mean).

At that audit, verification used Python 3.14.6. The environment virtualenv had
Gymnasium 1.3.0, NumPy 2.5.3, cloudpickle 3.1.2, and farama-notifications 0.0.6.

```bash
cd /Users/adityajain/CS/Projects/AIDEN
../.venv/bin/python -m unittest discover -s tests -v
python3 smoke_env.py
```

The suite at that date passed **52 tests**:

- 10 loader tests: complete and incomplete windows, split dates, duplicates,
  timestamp precision, causality, and malformed data.
- 28 execution/environment/action tests: hand-priced fills, all four action
  sizes, adaptive catch-up, final override, insufficient depth, fees, reward
  identities, and conservation across a seeded mixed-action path.
- 9 feature tests: hand calculations, future-data isolation, 12-interval rolling
  history, zero-volume/depth behavior, unit scaling, and validation.
- 5 Gym tests: official `check_env`, action rollouts, seeded and explicit episode
  selection, incomplete termination, and invalid calls.

Two regressions deserve special attention: a `1e-13 BTC` request against zero
depth remains unfilled; a `1 BTC` request against 25 levels of `0.04 BTC` fills
exactly without a false residual. The tests cover both errors that a generic
"close enough, call it filled" tolerance would conceal.

Reward checks establish that higher net proceeds improve reward for the same
parent, completed reward equals negative net shortfall, and withholding all
inventory cannot erase spread and fee costs from the conservative score. These
are checks of the chosen objective, not proof that it is the best research
objective for every order-size regime.

The Gym checker emits two warnings for intentionally unbounded Box dimensions.
Returns and log depth/volume ratios use fixed scales without arbitrary clipping.
Actual observations are required to be finite float32 arrays, and all exercised
observations were contained in the declared observation space.

The standard-library core can also be tested with system `python3`; the five
Gym tests are explicitly skipped when its dependency is absent. The venv
command above ran all 52 tests without skips at the audit date. Follow the
current [README](../README.md) for portable setup and verification commands.

## Real-data smoke check

One complete training episode, **2020-09-01 00:05:00 UTC**, was replayed with
`Q=0.01 BTC`, zero fees, and final liquidation enabled. Each run repeated one
constant action until completion. There is no random policy seed in these runs.

| Constant action | Decisions used | Completion | Net USDT | Completion score (bps) |
| --- | ---: | ---: | ---: | ---: |
| WAIT | 60 | 1.0 | 116.354400 | 9.981055 |
| BELOW_SCHEDULE | 60 | 1.0 | 116.385933 | 7.273650 |
| ON_SCHEDULE | 60 | 1.0 | 116.398245 | 6.216602 |
| ABOVE_SCHEDULE | 59 | 1.0 | 116.408413 | 5.343586 |

WAIT completes because the shared deadline rule submits the remaining parent
at the last book. This table verifies data-to-environment integration. It is
not a baseline study, order-size calibration, or evidence that one action is
generally superior. At the time of this smoke check, no policy had been
evaluated on the March test day.

For every processed day's historical integrity audit, episode counts, and raw-file
provenance, see [data_checks.md](data_checks.md).

## Completed-study status: 2026-09-29 UTC

The project subsequently completed three PPO training runs using the real
September--January episodes, full February model selection, and held-out March
evaluation. Each seed collected 200,704 training interactions. Seed 19 was
selected on February and its checkpoint identity was frozen before March
evaluation. The existing cleaned files were used unchanged.

The current suite additionally checks preprocessing boundaries, baseline
schedules, training/checkpoint behavior, paired result accounting, and model
selection guards. Synthetic fixtures test software behavior; all reported
research performance comes from the collected market data. The historical
single-episode smoke table above is not part of the performance comparison.

See [completed results](../README.md#15-what-the-measured-results-mean) for measured
costs, completion, tails, and limitations, and the [experiment protocol](experiment.md)
for the fixed training, selection, and evaluation procedure.
The detailed original report is `artifacts/RESULTS.md` in the optional local bundle.
