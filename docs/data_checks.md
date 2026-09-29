# Data checks and episode loading

The audit results below are the **historical 2026-09-21 integrity audit**.
The project subsequently completed PPO training, February selection, and March
evaluation on the same cleaned inputs. See the completed-study status at the
end of this document and [results](../README.md#15-what-the-measured-results-mean).

The inputs are seven separate BTCUSDT days, each the first of its month. They
are not continuous months. Validation and testing each represent one day.

## Loader contract

`data.load_episodes(path)` streams a processed `.csv` or `.csv.gz` and yields
lists containing exactly 60 rows. Each row is a dictionary: numeric fields are
floats, timestamps are timezone-aware UTC `datetime` values, and a missing
no-trade VWAP is `None`. No pandas dependency is needed.

`data.load_split()` defaults to training. Pass `"validation"` or `"testing"`
explicitly to access those splits. Filenames and expected dates are fixed in
`SPLIT_DAYS`; a wrong date fails instead of silently contaminating another split.

```python
from data import LoadStats, load_split

stats = LoadStats()
for episode in load_split("training", stats=stats):
    # Construct an environment for this episode and run a strategy here.
    pass
print(stats)  # Counters are final after exhausting the iterator.
```

Only one five-minute window is buffered. Timestamps must already be strictly
increasing: duplicate or reversed rows raise an error; the loader does not sort
away evidence of a data problem. Decisions must be on the five-second grid.
Book receipts must precede or equal their decision and be at most five seconds
old. Timestamp fractions with more than six digits are rejected: Python's
microsecond resolution must not silently truncate a future receipt onto a decision
boundary. All 25 levels on both sides and both trade-volume fields are required.

An incomplete window is discarded as a whole. No forward filling, interpolation,
or artificial liquidity is introduced. `LoadStats` reports rows seen, complete
episodes, observed incomplete windows, and discarded rows. Completely absent
windows have no input rows and are not included in `incomplete_windows`.

The loader checks the file structure, timestamps, and numeric conversion. The
environment's validation checks book ordering, amount constraints, and trade
flow consistency. Keeping these responsibilities separate avoids two execution
validators drifting apart.

## Historical integrity audit: 2026-09-21

Every processed row was checked, including rows in incomplete windows. All
seven files have the same 110-column schema. Checks found:

- All required 25 bid and 25 ask price/amount pairs are present and finite.
- Prices are positive; amounts are nonnegative. Bids strictly decrease and asks
  strictly increase. Every best ask exceeds its best bid.
- Decision times are unique, increasing, UTC, and on the five-second grid.
  Each file contains only its expected calendar day.
- Book receipt age is between zero and five seconds. Stored `book_age_ms`,
  midprice, and spread agree with recomputation within float tolerance.
- Trade counts are nonnegative integers, total volume and notional are
  nonnegative, and absolute signed volume does not exceed total volume within
  rounding tolerance.
- There were no zero-size best bids or asks in the audited processed files.

No violations were found in these checks. This is a format/accounting audit,
not a reconstruction of every raw trade aggregation or a claim of live-market
fidelity. At the time of this audit, the March file was checked only for data
integrity; no policy had yet been evaluated or tuned on it. That historical
statement does not describe the later completed study.

| Split | Day | Processed rows | Complete episodes | Observed incomplete windows | Discarded rows | Maximum book age (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Training | 2020-09-01 | 17,279 | 287 | 1 | 59 | 1,212.527 |
| Training | 2020-10-01 | 17,279 | 287 | 1 | 59 | 2,784.722 |
| Training | 2020-11-01 | 17,278 | 287 | 1 | 58 | 2,041.439 |
| Training | 2020-12-01 | 16,724 | 276 | 7 | 164 | 4,038.265 |
| Training | 2021-01-01 | 17,279 | 287 | 1 | 59 | 1,160.508 |
| Validation | 2021-02-01 | 17,273 | 285 | 3 | 173 | 4,827.495 |
| Testing | 2021-03-01 | 17,279 | 287 | 1 | 59 | 3,836.535 |

There are **1,424 training**, **285 validation**, and **287 testing** episodes,
or **1,996 total**. December has five entirely absent five-minute windows in
addition to its seven observed incomplete windows. The other days have no
entirely absent windows. November starts at 00:00:10 UTC; all other processed
days start at 00:00:05 UTC. All end at 23:59:55 UTC.

To reproduce the loader counts from the project directory:

```bash
python3 - <<'PY'
from pathlib import Path
from data import LoadStats, load_episodes
for path in sorted(Path("data").glob("*/*_5s.csv.gz")):
    stats = LoadStats()
    for episode in load_episodes(path, stats=stats):
        pass
    print(path, stats)
PY
```

## September raw-row discrepancy recorded in that audit

The file audited at
`data/training/Sep2020_T25.gz` was streamed in full, counting data rows after
the header. Its identity is:

- Size: **53,163,590 bytes** (compressed).
- Rows: **751,701**, excluding the header.
- SHA-256 of the compressed file:
  `e385b553f901500c5f5d7773b2ba1a5ff16b5a8fc1c882c8fc1cb064aeca9500`.
- First receipt: `2020-09-01T00:00:00.528977+00:00`.
- Last receipt: `2020-09-01T23:59:59.909722+00:00`.
- No decreasing receipt timestamps.

This count agrees with the notebook's saved `Book rows read: 751,701` output.
The earlier **5,206,603** claim has no matching file/path/hash or saved output
identified in that audit. It should not be used as a dataset fact.
The audited file's count is verified; the origin of the old claim remains
unresolved. No source or processed files were rewritten for this audit.

At the audit date, the notebook source already masked negative timestamp
sentinels before datetime conversion, so its saved midnight diagnostic could
reflect an older version. Its trailing undefined `main()` call also prevented
a clean rerun. These were observations about the notebook at that date.
The notebook is preserved unchanged locally at
`.local/archive/dataprocess.ipynb`; its saved market output is excluded from Git.
[preprocess.py](../preprocess.py) is the canonical preprocessing entry point.

## Completed-study provenance: 2026-09-29 UTC

The completed study used the existing collected and cleaned files unchanged;
it did not regenerate the seven days. Calibration used only the five training
days. The frozen configuration records SHA-256 hashes of those training files,
and the study artifacts record the model and configuration identities used
for evaluation.

Three PPO seeds were trained on the 1,424 training episodes. Checkpoints were
selected using February, then the winning seed was frozen before evaluating
all 287 complete March episodes. March results are now reported; they were
not used to select a seed or retrain a policy. The original seven-day coverage
and missing-window limitations still apply.

See [completed results](../README.md#15-what-the-measured-results-mean), the
[experiment protocol](experiment.md), and [local data setup](data_setup.md).
The detailed original report is available with the optional local bundle at
`artifacts/RESULTS.md`.
