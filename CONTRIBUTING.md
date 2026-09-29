# Working on AIDEN

Use the [documentation map](docs/README.md) to find a module or guide, and the
[main README](README.md) for the implementation and research assumptions.

## Set up and validate

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The recorded full experiment used Python 3.14.6. The full requirements install
the training and reporting dependencies. The smaller `requirements-env.txt` is
only for Gym adapter work; it is not sufficient for the complete test suite.

Implementation tests use controlled fixtures and do not need the local market
data. Research training/evaluation does need the collected inputs described in
[data setup](docs/data_setup.md). Never substitute fixture or smoke-test scores
for historical performance results.

## Keep changes reviewable

- Keep runtime modules at their documented paths unless a deliberate refactor
  also updates imports, commands, and the experiment lineage.
- Preserve exact units, action IDs, feature order/scaling, and accounting
  contracts when working on the execution interface.
- Run the checks relevant to a behavior change. Documentation-only changes need
  working links and valid examples, not a new full training run.
- Describe what changed, why, and how it was checked. Include any effect on
  causality, completion, fees, or interpretation of results.
- Use the existing tests for small, hand-checkable cases. New raw market extracts
  do not belong in a test fixture or notebook output.

The editor configuration describes whitespace conventions; it does not ask for
repository-wide reformatting. Text attributes keep new Git text checkouts on LF
line endings without modifying the frozen source during this cleanup.

## Preserve research provenance

Use a new `runs/` directory for each experiment. Keep the original configuration,
selected checkpoints, manifests, and completed outputs intact. Changing frozen
source or inputs may intentionally fail an integrity check; record a new study
instead of replacing an old digest to make the check pass.

Calibrate only from training data, choose policies on validation, and reserve new
test data for the declared final comparison. The original March day is already
inspected; another run on it is a reproduction or follow-up analysis, not a new
untouched holdout. See the [experiment protocol](docs/experiment.md).

## Keep local files out of Git

The default source-only publication boundary is:

| Commit | Keep local |
|---|---|
| Python source and tests | `data/`, raw/cleaned tables, market extracts |
| `configs/experiment.json` and dependency files | `runs/`, `artifacts/`, checkpoints and serialized models |
| README, focused documentation, CI, editor settings | Logs, caches, environments, scratch work and credentials |
| Small illustrative values written into test code | Exploratory notebooks containing saved market output |

The archival notebook is under `.local/archive/` and remains unchanged. It is
not required to run the project. New notebooks should not include market rows,
credentials, or machine-specific outputs in a proposed commit.

Common data/model formats are ignored even outside the usual local directories.
If a small non-market fixture or document asset ever needs an exception, review
its contents and add a narrow, explicit ignore exception. Do not force-add an
entire ignored directory. `.env.example` and `.env.sample` may contain placeholder
values only; a filename does not make real credentials safe to commit.

Before committing, inspect the actual Git index:

```bash
git status --short
git diff --check
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only --diff-filter=ACMR
git ls-files -- data runs artifacts .local dataprocess.ipynb
git ls-files --cached --ignored --exclude-standard
```

The final two commands should produce no paths. The `ACMR` filter shows added,
copied, modified, and renamed entries; staged deletions of formerly tracked
outputs are expected during cleanup and do not mean the local outputs vanished.

For a suspicious path, `git check-ignore -v PATH` explains its ignore rule.
`.gitignore` prevents ordinary accidental additions, but it does not inspect
arbitrary file contents, override `git add -f`, or remove old Git history.
Inspect notebook outputs and newly added files as well as their names.

If a file already exists in an older commit, a later deletion removes it only
from the current tree. A history purge requires a separate, coordinated history
rewrite; a normal cleanup commit cannot retroactively remove it.
