# Local data setup

The project uses seven isolated BTCUSDT days, each the first day of its month.
They are not seven continuous months. The collected data are local inputs;
this repository does not download them or infer account credentials. Run the
commands below from the project root with the dependencies in
[requirements.txt](../requirements.txt) installed in your Python environment.

| Split directory | Day labels | Calendar days |
| --- | --- | --- |
| `data/training/` | `Sep2020`, `Oct2020`, `Nov2020`, `Dec2020`, `Jan2021` | 2020-09-01 through 2021-01-01, first days only |
| `data/validation/` | `Feb2021` | 2021-02-01 |
| `data/testing/` | `Mar2021` | 2021-03-01 |

These memberships are fixed in `data.SPLIT_DAYS` and mirrored by the
preprocessing CLI. The loader checks each file's expected calendar day.

## Inputs and filenames

For each day label, place the two original gzip-compressed CSV exports in its
split directory. For example:

```text
data/training/Sep2020_T25.gz
data/training/Sep2020_Trade.gz
```

The book export requires `local_timestamp` plus all 25 price/amount levels on
both sides: `bids[0].price`, `bids[0].amount`, …, `asks[24].price`,
`asks[24].amount`. The trade export requires `local_timestamp`, `price`,
`amount`, and `side`. Raw local timestamps are integer UTC microseconds; prices
are USDT/BTC and amounts are BTC. Trade sides are `buy`, `sell`, or `unknown`;
unknown trades contribute volume but zero signed volume. Book rows must be in
nondecreasing receipt-time order.

Cleaned outputs follow the same day labels:

```text
data/training/Sep2020_5s.csv.gz
data/validation/Feb2021_5s.csv.gz
data/testing/Mar2021_5s.csv.gz
```

If these cleaned files are already available, the environment and study can
load them directly. **The completed study used the existing collected and
cleaned files unchanged**; it did not rerun preprocessing.

## Preprocessing raw exports

[preprocess.py](../preprocess.py) is the canonical preprocessing entry point.
The original notebook is preserved unchanged locally at
`.local/archive/dataprocess.ipynb`. Its saved market output is excluded from Git,
and it is not required to reproduce the pipeline.
To generate missing cleaned files from your raw exports, run each split
explicitly:

```bash
python preprocess.py --split training
python preprocess.py --split validation
python preprocess.py --split testing
```

The default split is training. Restrict work to one day with
`--split training --month Sep2020`. Use `--data-dir /path/to/data` to change the
data root while retaining the three split subdirectories. Outputs are written
beside their raw inputs. Existing outputs cause an error by default; replacing
them requires an explicit `--overwrite` flag.

For each five-second decision, preprocessing selects the most recently
received book available at that time. Trades are aggregated over the preceding
interval `(t - 5 seconds, t]`. Missing books and books older than five seconds
are dropped; invalid depth, ordering, or crossed quotes cause errors. No
future book or trade enters the decision features.

The loader then keeps only complete, UTC-aligned five-minute windows of 60
decisions. It discards incomplete windows instead of filling gaps. The current
files yield **1,424 training, 285 validation, and 287 testing episodes**.
See [data checks](data_checks.md) for counts and missing-window details, the
[experiment protocol](experiment.md) for split usage, and
[completed results](../README.md#15-what-the-measured-results-mean) for the measured study.
The detailed report and result tables are in the optional local `artifacts/` bundle.
