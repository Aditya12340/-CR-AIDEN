"""Copy a completed study into a new, hash-verified publication directory.

Only explicit result directories are copied. Raw/clean market inputs, caches,
credentials, and unrelated experiments are never part of the bundle.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from select_model import sha256


PROJECT = Path(__file__).resolve().parent
RESULT_FILES = ("RESULTS.md", "frozen_selection.json")
RESULT_DIRS = ("ppo", "training", "validation", "testing", "robustness", "figures")


def package(run: Path, output: Path) -> dict:
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or run in output.parents or output in run.parents:
        raise ValueError("bundle and run must be separate directory trees")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    for name in (*RESULT_FILES, *RESULT_DIRS):
        if not (run / name).exists():
            raise FileNotFoundError(f"incomplete study: missing {run / name}")
    frozen = json.loads((run / "frozen_selection.json").read_text())
    for name, digest in frozen["source_sha256"].items():
        if sha256(PROJECT / name) != digest:
            raise ValueError(f"source changed since the experiment froze: {name}")
    for name, checkpoint in frozen["checkpoints"].items():
        seed = int(name.removeprefix("ppo_seed_"))
        if sha256(run / "ppo" / f"seed_{seed}" / "model.zip") != checkpoint["sha256"]:
            raise ValueError(f"checkpoint changed: {name}")
    if sha256(run / "validation" / "per_episode.csv") != frozen["validation_results_sha256"]:
        raise ValueError("validation result file changed since selection")

    output.mkdir(parents=True)
    for name in RESULT_FILES:
        shutil.copy2(run / name, output / name)
    for name in RESULT_DIRS:
        shutil.copytree(run / name, output / name)

    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(output)),
                          "bytes": path.stat().st_size, "sha256": sha256(path)})
    sources = [*PROJECT.glob("*.py"), *PROJECT.glob("requirements*.txt"),
               *PROJECT.glob("configs/*.json"), *PROJECT.glob("tests/*.py")]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "origin_run": str(run),
        "selected_policy": frozen["selected"]["policy"],
        "selected_checkpoint": f"ppo/seed_{frozen['selected']['seed']}/model.zip",
        "market_data_included": False,
        "provenance_note": "Original logs retain their execution-time absolute paths. "
                           "Bundle file paths below are relative and portable.",
        "files": files,
        "source_sha256": {str(path.relative_to(PROJECT)): sha256(path)
                          for path in sorted(sources)},
    }
    (output / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Packaged {len(files)} files, {sum(row['bytes'] for row in files)/1e6:.1f} MB: {output}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.run, args.output)


if __name__ == "__main__":
    main()
