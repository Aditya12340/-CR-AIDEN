# Documentation map

Start with the [main README](../README.md) for the complete explanation. Its
[contents](../README.md#contents) link to all 20 sections; the table below gives
shorter routes for common tasks.

| I want to… | Read this |
|---|---|
| Understand the project and measured outcome | [Project overview](../README.md#1-the-project-in-one-page) and [results](../README.md#15-what-the-measured-results-mean) |
| Find the module responsible for a behavior | [Architecture and module map](../README.md#3-architecture-and-repository-map) |
| Set up the actual historical inputs | [Data setup](data_setup.md) and [data audit](data_checks.md) |
| Understand fills, actions, features, and reward | [Environment design](environment.md) and [execution walkthrough](../README.md#7-the-execution-environment) |
| Understand PPO and checkpoint selection | [Training explanation](../README.md#12-the-neural-networks-and-ppo-training) and [experiment protocol](experiment.md) |
| Interpret TWAP/VWAP comparisons | [Strategies](../README.md#11-the-comparison-strategies), [evaluation](../README.md#14-how-evaluation-is-calculated), and [VWAP guide](vwap.md) |
| Install dependencies and reproduce a study | [Installation and reproduction](../README.md#17-installation-and-reproduction) |
| Understand what was tested | [Test map](../README.md#18-tests-edge-cases-and-troubleshooting) and [completed-study verification](verification.md) |
| Inspect the earlier environment work | [Historical environment validation](environment_validation.md) |
| Make a change or prepare a commit | [Contribution guide](../CONTRIBUTING.md) |

## Where files belong

```text
AIDEN/
├── README.md               # full project guide and measured results
├── CONTRIBUTING.md         # development and commit workflow
├── *.py                    # execution, learning, and experiment modules
├── configs/                # reproducible experiment configuration
├── tests/                  # controlled implementation tests
├── docs/                   # focused guides and verification records
├── .github/workflows/      # automated implementation tests
├── requirements*.txt       # dependency specifications
├── data/                   # LOCAL: collected and cleaned market inputs
├── runs/                   # LOCAL: working experiments
├── artifacts/              # LOCAL: completed checkpoints and result bundle
└── .local/archive/         # LOCAL: preserved exploratory notebook
```

The Python modules intentionally remain at the repository root. Their imports,
documented CLI commands, and frozen experiment manifests use those paths. The
module map organizes them by responsibility without changing the implementation.

The original `dataprocess.ipynb` is preserved unchanged locally at
`.local/archive/dataprocess.ipynb`. Its saved outputs include market rows, its
code contains machine-specific paths, and it is archival rather than a supported
entry point. Use [preprocess.py](../preprocess.py) for executable preprocessing.

## What a GitHub checkout includes

Source, tests, configuration, dependency specifications, CI, and documentation
belong in Git. Raw/cleaned inputs, generated CSVs, model weights, experiment
bundles, logs, caches, credentials, and the exploratory notebook remain local.
The [ignore rules](../.gitignore) also cover common data and checkpoint formats
if they are accidentally copied outside their normal directories.

The results are explained in the main README. References to `artifacts/...`
name optional local evidence, not files promised by a source-only checkout.
Historical audit documents retain their original dates and findings; use
[verification.md](verification.md) for the completed-study verification history.

Ignore rules do not erase earlier Git commits. Removing previously committed
files from the current tree and purging them from shared history are different
operations.
