"""Freeze the PPO checkpoint chosen on full February validation, before testing."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_model(summary, config):
    """Minimize all-episode completion score, not survivor-only execution cost."""
    metadata = summary["metadata"]
    if (metadata["split"] != "validation" or metadata.get("diagnostic_subset")
            or metadata.get("max_episodes") is not None):
        raise ValueError("selection requires the full validation split")
    if metadata["episode_count"] <= 0:
        raise ValueError("validation must contain complete episodes")
    load_stats = metadata.get("load_stats")
    if load_stats and load_stats["complete_episodes"] != metadata["episode_count"]:
        raise ValueError("validation results do not cover every loaded episode")
    if metadata["fee_bps"] != config["fee_bps"]:
        raise ValueError("validation fees differ from the frozen configuration")
    candidates = []
    for seed in config["seeds"]:
        name = f"ppo_seed_{seed}"
        groups = [group for group in summary["groups"]
                  if group["policy"] == name and group["size"] == config["primary_size"]]
        if len(groups) != 1:
            raise ValueError(f"missing or duplicate primary-size validation for {name}")
        group = groups[0]
        if (group["fee_bps"] != config["fee_bps"] or
                group["parent_quantity_btc"] != config["sizes"][config["primary_size"]]):
            raise ValueError("candidate fee or quantity differs from the configuration")
        if group["episodes"] != metadata["episode_count"]:
            raise ValueError("every model must be evaluated on every validation episode")
        score = group["completion_score_bps"]
        if score["n"] != group["episodes"] or not math.isfinite(score["mean"]):
            raise ValueError("selection score must include every episode and be finite")
        if not math.isfinite(group["completion_rate"]) or not 0 <= group["completion_rate"] <= 1:
            raise ValueError("completion rate must be finite and in [0, 1]")
        candidates.append({"seed": seed, "policy": name,
                           "mean_completion_score_bps": group["completion_score_bps"]["mean"],
                           "completion_rate": group["completion_rate"]})
    ordered = sorted(candidates, key=lambda row: (row["mean_completion_score_bps"], row["seed"]))
    return ordered[0], ordered


def freeze_selection(config_path, validation_dir, models_dir, output_path):
    config_path, validation_dir, models_dir = map(Path, (config_path, validation_dir, models_dir))
    config = json.loads(config_path.read_text())
    summary = json.loads((validation_dir / "summary.json").read_text())
    if summary["metadata"]["config_sha256"] != sha256(config_path):
        raise ValueError("configuration changed since validation")
    selected, candidates = choose_model(summary, config)
    checkpoints = {}
    for row in candidates:
        path = models_dir / f"seed_{row['seed']}" / "model.zip"
        digest = sha256(path)
        evaluated = summary["metadata"]["models"][row["policy"]]["sha256"]
        if digest != evaluated:
            raise ValueError(f"checkpoint changed after validation: {path}")
        checkpoints[row["policy"]] = {"path": str(path), "sha256": digest}
    source_files = ("env.py", "features.py", "gym_env.py", "train.py", "data.py",
                    "baselines.py", "calibrate.py", "evaluate.py", "reporting.py")
    record = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation", "selection_day": "2021-02-01",
        "selection_episodes": summary["metadata"]["episode_count"],
        "criterion": "minimum mean completion_score_bps at primary size; lower seed breaks ties",
        "selected": selected, "candidates": candidates, "checkpoints": checkpoints,
        "config_sha256": sha256(config_path), "config": config,
        "validation_results_sha256": sha256(validation_dir / "per_episode.csv"),
        "source_sha256": {name: sha256(PROJECT / name) for name in source_files},
        "heldout_protocol": "Evaluate all three saved seeds on March at all three fixed sizes; primary result is selected seed. No selection or retraining on March.",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Frozen {selected['policy']} from {record['selection_episodes']} validation episodes: "
          f"{selected['mean_completion_score_bps']:.6f} bps", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze_selection(args.config, args.validation, args.models_dir, args.output)


if __name__ == "__main__":
    main()
