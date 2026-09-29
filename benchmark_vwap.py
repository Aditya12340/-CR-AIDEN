"""Add a real-data VWAP benchmark to an already frozen, packaged PPO study.

This supplement reuses the original PPO/TWAP fills after checking their hashes.
Only the new historical-volume execution schedule is replayed. The trade VWAP
price is an ex-post measurement and never enters the schedule or PPO policy.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from math import fsum
from pathlib import Path
import sys

from calibrate import episode_market_context
from data import LoadStats, load_split
from env import SellExecutionEnv, STEP_SECONDS
from evaluate import data_manifest, installed_versions, regime_label, write_csv
from robustness import read_results
from run_experiment import verify_training_inputs
from select_model import sha256
from vwap import fit_volume_profile, market_vwap, schedule_weights, vwap_request
from vwap_reporting import (add_market_metrics, compare_and_summarize,
                            make_figures, write_report)


PROJECT = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_study(study):
    """Check archived fills/models and the exact execution code used originally."""
    study = Path(study)
    manifest = json.loads((study / "bundle_manifest.json").read_text())
    for entry in manifest["files"]:
        path = study / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"original artifact changed: {entry['path']}")
    frozen = json.loads((study / "frozen_selection.json").read_text())
    for name, digest in frozen["source_sha256"].items():
        if sha256(PROJECT / name) != digest:
            raise ValueError(f"original execution/evaluation source changed: {name}")
    if sha256(study / "validation" / "per_episode.csv") != frozen["validation_results_sha256"]:
        raise ValueError("original validation results differ from frozen selection")
    selected = frozen["selected"]["policy"]
    checkpoint = study / "ppo" / f"seed_{frozen['selected']['seed']}" / "model.zip"
    if sha256(checkpoint) != frozen["checkpoints"][selected]["sha256"]:
        raise ValueError("selected checkpoint differs from frozen selection")
    return frozen


def run_vwap(episode, quantity, size, split, config, weights):
    """Use the shared book sweep/accounting and export the existing result schema.

    Only record construction is repeated here to keep the original frozen
    experiment source byte-for-byte intact. No alternate execution engine is used.
    """
    env = SellExecutionEnv(episode, quantity, fee_bps=config["fee_bps"])
    while not env.done:
        env.step(vwap_request(env, weights))
    context = episode_market_context(env.episode)
    thresholds = config["market_regimes"]
    summary = asdict(env.summary())
    start = env.episode[0]["decision_time"]
    identifiers = {
        "episode_id": start.isoformat(), "episode_start": start.isoformat(),
        "day": start.date().isoformat(), "split": split, "policy": "vwap",
        "size": size, "parent_quantity_btc": float(quantity),
        "fee_bps": float(config["fee_bps"]),
    }
    filled = summary["filled_btc"]
    trade_decisions = sum(step.filled_btc > 0 for step in env.history)
    record = {
        **identifiers, **summary, **context,
        "spread_regime": regime_label(context["arrival_spread_bps"], thresholds["arrival_spread_bps"]),
        "volatility_regime": regime_label(context["episode_realized_volatility_bps"],
                                          thresholds["episode_realized_volatility_bps"]),
        "average_execution_time_seconds": (
            fsum(step.filled_btc * step.step_index * STEP_SECONDS for step in env.history) / filled
            if filled > 0 else None),
        "last_decision_seconds": env.history[-1].step_index * STEP_SECONDS,
        "decisions": len(env.history), "trade_decisions": trade_decisions,
        "mean_trade_size_btc": filled / trade_decisions if trade_decisions else None,
        "forced_terminal_liquidation": any(step.forced_terminal_liquidation for step in env.history),
        "final_decision_quantity_fraction": fsum(step.filled_btc for step in env.history
                                                  if step.step_index == 59) / quantity,
        **{f"action_{action}_count": 0 for action in range(4)},
    }
    trace = []
    for step in env.history:
        details = asdict(step)
        details["decision_time"] = step.decision_time.isoformat()
        trace.append({**identifiers, "elapsed_seconds": step.step_index * STEP_SECONDS,
                      "forecast_weight": weights[step.step_index], **details})
    return record, trace


def archived_records(study, split, selected, config):
    """Index saved policies by exact obligations; reject changed fees or quantities."""
    rows = [row for row in read_results(study / split / "per_episode.csv")
            if row["policy"] in (selected, "twap")]
    indexed = {}
    for row in rows:
        if (row["split"] != split or row["size"] not in config["sizes"]
                or row["parent_quantity_btc"] != config["sizes"][row["size"]]
                or row["fee_bps"] != config["fee_bps"]):
            raise ValueError("archived result does not match the configured obligation")
        key = (row["episode_id"], row["size"], row["policy"])
        if key in indexed:
            raise ValueError(f"duplicate archived result: {key}")
        indexed[key] = row
    return indexed


def benchmark(study, output):
    study, output = Path(study).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    frozen = verify_study(study)
    config, selected = frozen["config"], frozen["selected"]["policy"]
    data_dir = verify_training_inputs(config)
    original_metadata = {}
    inputs = {}
    # Verify all split data before fitting or scoring anything. Original saved
    # cash may only be reused against the identical historical market files.
    for split in ("training", "validation", "testing"):
        original = json.loads((study / split / "run_metadata.json").read_text())
        if original["config_sha256"] != frozen["config_sha256"]:
            raise ValueError(f"{split}: archived configuration differs from selection")
        if original["models"][selected]["sha256"] != frozen["checkpoints"][selected]["sha256"]:
            raise ValueError(f"{split}: saved selected-policy results use another checkpoint")
        current = data_manifest(split, data_dir)
        identity = lambda entries: sorted((Path(row["path"]).name, row["sha256"]) for row in entries)
        if identity(current) != identity(original["data_manifest"]):
            raise ValueError(f"{split}: market data differs from the archived evaluation")
        original_metadata[split] = original
        inputs[split] = {"data_manifest": current,
                         "archived_results_sha256": sha256(study / split / "per_episode.csv")}

    output.mkdir(parents=True)
    print("Fitting historical volume profile on the actual training days only.", flush=True)
    profile = fit_volume_profile(load_split("training", data_dir))
    write_json(output / "volume_profile.json", profile)
    source_names = ("vwap.py", "benchmark_vwap.py", "vwap_reporting.py")
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_policy": selected, "config": config,
        "primary_size": config["primary_size"],
        "original_study": str(study), "original_frozen_at_utc": frozen["frozen_at_utc"],
        "original_selection_sha256": sha256(study / "frozen_selection.json"),
        "selected_model_sha256": frozen["checkpoints"][selected]["sha256"],
        "profile_sha256": sha256(output / "volume_profile.json"),
        "source_sha256": {name: sha256(PROJECT / name) for name in source_names},
        "profile_training_episodes": profile["episodes"],
        "profile_training_rows": profile["rows"],
        "inputs": inputs,
        "method": "Arithmetic mean historical trailing-five-second BTC volume by UTC decision minute; "
                  "fixed minute buckets pool up to 12 observations per day. Missing buckets use the "
                  "global training mean. At each decision request remaining BTC times the current "
                  "forecast weight divided by remaining forecast weights; use TWAP if all are zero.",
        "market_benchmark": "Receipt-time trade notional / volume over (arrival, arrival+295s], "
                            "59 bins, excluding the prearrival bin; fixed horizon even for early completion.",
        "discretization": "60 execution weights describe expected local activity at decision times. "
                          "They do not represent 60 observed postarrival trade intervals.",
        "scope": "Supplementary benchmark requested after the original March results were known. "
                 "Forecast fit only on training; no PPO retraining or reselection. Historical profile "
                 "training-split results are in-sample diagnostics. Not an institutional VWAP replica.",
        "saved_policy_reuse": "Original selected PPO and TWAP execution records are reused exactly "
                              "after hash validation; neither policy is rerun or retuned.",
        "command": [sys.executable, *sys.argv],
    }
    # Freeze the new forecast and method before calculating the new comparisons.
    write_json(output / "protocol.json", metadata)
    write_json(output / "versions.json", installed_versions())
    all_records, traces, profile_diagnostics = [], [], []
    for split in ("training", "validation", "testing"):
        archived = archived_records(study, split, selected, config)
        seen = set()
        stats = LoadStats()
        split_records = []
        for episode_index, episode in enumerate(load_split(split, data_dir, stats=stats)):
            start = episode[0]["decision_time"]
            weights = schedule_weights(profile, start)
            total = fsum(weights)
            proportions = [weight / total if total else 1 / 60 for weight in weights]
            profile_diagnostics.append({
                "split": split, "episode_id": start.isoformat(),
                "max_slice_fraction": max(proportions),
                "min_slice_fraction": min(proportions),
                "l1_distance_from_twap": fsum(abs(weight - 1 / 60) for weight in proportions),
            })
            market = market_vwap(episode)  # Reporting only; never passed into vwap_request.
            for size, quantity in config["sizes"].items():
                for policy in (selected, "twap"):
                    key = (start.isoformat(), size, policy)
                    if key not in archived:
                        raise ValueError(f"missing archived policy result: {key}")
                    seen.add(key)
                    split_records.append(add_market_metrics(archived[key], market))
                record, steps = run_vwap(episode, quantity, size, split, config, weights)
                split_records.append(add_market_metrics(record, market))
                if episode_index == 0:
                    traces.extend(steps)
            if (episode_index + 1) % 100 == 0:
                print(f"Replayed VWAP on {episode_index + 1} {split} episodes.", flush=True)
        if set(archived) != seen or stats.complete_episodes != original_metadata[split]["episode_count"]:
            raise ValueError(f"{split}: actual episode coverage does not match original results")
        inputs[split]["load_stats"] = asdict(stats)
        all_records.extend(split_records)
        print(f"Completed {split}: {stats.complete_episodes} actual episodes, all three sizes.", flush=True)

    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    pairs, summary = compare_and_summarize(all_records, selected, metadata)
    write_csv(output / "per_episode.csv", all_records)
    write_csv(output / "paired_results.csv", pairs)
    write_csv(output / "first_vwap_episode_steps.csv", traces)
    write_csv(output / "profile_diagnostics.csv", profile_diagnostics)
    write_json(output / "summary.json", summary)
    write_json(output / "run_metadata.json", metadata)
    write_report(summary, output / "REPORT.md")
    make_figures(all_records, summary, output / "figures")
    files = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
              "sha256": sha256(path)} for path in sorted(output.rglob("*")) if path.is_file()]
    write_json(output / "manifest.json", {"files": files, "source_sha256": metadata["source_sha256"],
                                          "original_selection_sha256": metadata["original_selection_sha256"]})
    print(f"Saved {len(all_records)} actual paired policy records to {output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=PROJECT / "artifacts",
                        help="packaged original study with bundle_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.study, args.output)


if __name__ == "__main__":
    main()
