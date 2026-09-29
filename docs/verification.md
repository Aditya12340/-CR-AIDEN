# Completed-study verification — 2026-09-29

The [measured results](../README.md#15-what-the-measured-results-mean) summarize
the study. The detailed local report, `artifacts/RESULTS.md`, and its CSVs are the
underlying performance evidence; that optional bundle is excluded from Git.
The tests below check implementation behavior separately.

## Test suite

From the repository root, with the full pinned dependencies installed:

```bash
python -m unittest discover -s tests -v
```

The final full suite passed **104 tests, with no skips**, in 2.130 seconds on
Python 3.14.6/macOS. Coverage includes book sweeps, inventory/cash conservation,
deadline failures, causal features, date splits, preprocessing boundaries,
Gymnasium's official checker, baseline behavior, real PPO parameter updates and
checkpoint reload, validation-only selection, paired metrics, fee repricing,
and the configured data-directory integrity guard.

Gymnasium warns that some observation bounds are infinite. The features have
finite-value checks; the unbounded space intentionally avoids inventing fixed
maximum returns, spreads, or volumes. The checker passed.

The small PPO unit test uses a controlled fixture. Its printed validation score
is not a research result. Full training used the collected data and produced
the separate evidence below. The GitHub workflow runs implementation tests; it
does not have access to local market inputs or rerun the historical study.

## Actual training and evaluation

All three seeds completed 200,704 interactions, totaling **602,112**. Each
manifest records changed parameter hashes, selected checkpoint step, package
versions, and the actual validation indices. Training sampled from the 1,424
complete historical training episodes with replacement.

| Output | Episodes | Policies | Sizes | Saved result rows |
|---|---:|---:|---:|---:|
| Training diagnostic | 1,424 | 6 | 3 | 25,632 |
| Full February validation | 285 | 8 | 3 | 6,840 |
| Frozen March test | 287 | 8 | 3 | 6,888 |
| Fee scenarios from March fills | Same 287 | Same 8 | Same 3 | 20,664 across 3 fees |

The six training policies are five fixed controls plus selected seed 19; the
eight validation/test policies include all three PPO seeds. Fee scenarios reuse
recorded fills and are not additional independent episodes or retrained agents.

Training finished at **03:57:50 UTC**. Full February validation then selected
seed 19; selection froze at **04:02:07 UTC**, before March evaluation began at
**04:02:37 UTC**. March outcomes did not change the model or its hyperparameters.
The stages were run individually; `run_experiment.py` now orchestrates that same
sequence, with a tested input-hash guard. The wrapper was not used to rerun the
completed study merely to duplicate the results.

## Artifact audit

Independent reads confirmed that:

- All five training input hashes, frozen source hashes, and checkpoint hashes
  match their recorded files.
- Every split contains the intended calendar days and complete-episode counts.
- Recomputing means and completion counts from CSV reproduces saved summaries.
- Across 6,888 test result rows, inventory conservation error was at most
  2.7e-15 BTC; gross minus fees equals net cash; reward plus completion score
  differed by at most 8.8e-12 bps.
- Repricing the recorded fills at the original 10-bp fee reconstructs net cash
  within 3e-11 USDT of the original accounting.
- Learning curves and the cost/completion/fee figures were visually inspected.

`package_results.py` checks frozen source, model, and validation-result hashes
before copying the completed study. `artifacts/bundle_manifest.json` records a
SHA-256 and byte count for every bundled file. Absolute paths in original logs
are preserved as provenance; bundle paths in that manifest are relative.

These checks establish the recorded pipeline's accounting and provenance, not
live execution realism or significance beyond one test day. The historical
[environment verification](environment_validation.md) is retained separately.

## VWAP supplement

The subsequent VWAP extension expanded the suite to **128 passing tests with
no skips** (2.143 seconds on the same environment). The 24 additional checks
cover training-only profile fitting, missing-volume fallback, causal schedules,
weighted trade-price arithmetic, prearrival exclusion, incomplete orders,
gross/net slippage signs, exact archive pairing, and equal-weight equivalence
to the original TWAP execution path.

The extension lives in `vwap.py`, `benchmark_vwap.py`, and `vwap_reporting.py`.
The original frozen modules and research outputs retain their original hashes.
The local supplementary report, `artifacts/vwap/REPORT.md`, and its own manifest
record the additional full-data comparison. The [VWAP guide](vwap.md) contains
the measured summary, scope, and reproduction command for a source-only checkout.
