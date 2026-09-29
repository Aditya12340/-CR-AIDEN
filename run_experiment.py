"""Run the real-data research protocol in a new output directory.

This is intentionally explicit: train -> full validation -> freeze -> training
diagnostics -> held-out test -> robustness/report. It never calibrates on test.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from data import SPLIT_DAYS
from select_model import freeze_selection, sha256


PROJECT = Path(__file__).resolve().parent


def verify_training_inputs(config, *, project=PROJECT):
    """Verify every configured training file at the location training will read.

    Calibration records portable labels such as ``data/training/Sep2020_5s.csv.gz``.
    Their ``data/`` prefix denotes the configured data root, not necessarily the
    project's default directory. Exact manifest coverage prevents an omitted or
    duplicated entry from weakening the check.
    """
    project = Path(project)
    data_dir = Path(config.get("data_dir", project / "data"))
    if not data_dir.is_absolute():
        data_dir = project / data_dir

    expected = {
        Path("data") / "training" / f"{day}_5s.csv.gz"
        for day in SPLIT_DAYS["training"]
    }
    manifest = config.get("training_manifest")
    if not isinstance(manifest, list):
        raise ValueError("training_manifest must contain one entry for every training day")
    entries = {}
    for entry in manifest:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("every training manifest entry needs a portable path")
        label = Path(entry["path"])
        if label not in expected:
            raise ValueError(f"unexpected training manifest path: {entry['path']}")
        if label in entries:
            raise ValueError(f"duplicate training manifest path: {entry['path']}")
        if not isinstance(entry.get("sha256"), str):
            raise ValueError(f"training manifest entry needs a SHA-256: {entry['path']}")
        entries[label] = entry
    missing = expected.difference(entries)
    if missing:
        raise ValueError("training manifest is missing: " + ", ".join(str(path) for path in sorted(missing)))

    for label, entry in entries.items():
        path = data_dir / label.relative_to("data")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"training dataset differs from calibration: {path}")
    return data_dir


def run(command):
    print("Running:", " ".join(map(str, command)), flush=True)
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(PROJECT / ".mplconfig"))
    subprocess.run([str(part) for part in command], cwd=PROJECT, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/experiment.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = json.loads(config_path.read_text())
    verify_training_inputs(config)
    output.mkdir(parents=True, exist_ok=False)
    python = sys.executable
    run([python, "train.py", "--config", config_path, "--output", output / "ppo"])
    models = []
    for seed in config["seeds"]:
        models.extend(["--model", f"ppo_seed_{seed}={output / 'ppo' / f'seed_{seed}' / 'model.zip'}"])

    def evaluate(split, extra_models):
        run([python, "evaluate.py", "--split", split, "--config", config_path,
             "--output", output / split, *extra_models])

    evaluate("validation", models)
    frozen = freeze_selection(config_path, output / "validation", output / "ppo",
                              output / "frozen_selection.json")
    selected = frozen["selected"]["policy"]
    selected_model = frozen["checkpoints"][selected]["path"]
    evaluate("training", ["--model", f"{selected}={selected_model}"])
    # None of the commands above opens March for policy outcomes. All test
    # seeds/sizes are prespecified; the primary model cannot change afterward.
    evaluate("testing", models)
    run([python, "robustness.py", "--results-dir", output / "testing",
         "--config", config_path, "--selected-policy", selected,
         "--output", output / "robustness"])
    run([python, "study_report.py", "--run", output])
    print(f"Completed study: {output / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
