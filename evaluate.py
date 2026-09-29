"""Evaluate fixed baselines and saved PPO policies on exactly paired episodes.

This script never trains or chooses a checkpoint. Validation may guide the
separate selection step; invoke the testing split only after choices are frozen.
Every named policy sees the same parent size, episode, fee, and terminal rule.
"""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
from math import fsum, isfinite
from pathlib import Path
import platform
import sys

from baselines import BASELINE_NAMES, baseline_request
from calibrate import episode_market_context
from data import DATA_DIR, SPLIT_DAYS, LoadStats, load_split
from env import SellExecutionEnv, STEP_SECONDS
from features import observation_vector
from reporting import make_figures, paired_results, summarize_results, write_report


def regime_label(value, cutoffs):
    """Apply two fixed training cutoffs, keeping tied thresholds deterministic."""
    if cutoffs is None:
        return "unclassified"
    lower, upper = float(cutoffs["lower"]), float(cutoffs["upper"])
    if not all(isfinite(item) for item in (value, lower, upper)) or lower > upper:
        raise ValueError("regime values and ordered cutoffs must be finite")
    return "low" if value <= lower else "medium" if value <= upper else "high"


def evaluate_episode(episode, policy, quantity_btc, *, size, split,
                     fee_bps=10.0, model=None, market_regimes=None, context=None):
    """Run one deterministic policy, returning full metrics and actual step logs.

    A model uses four discrete urgency IDs; baselines request BTC directly. The
    observation vector is constructed identically to the Gym adapter. The full
    episode volatility in context is used only in the result after execution.
    """
    env = SellExecutionEnv(episode, quantity_btc, fee_bps=fee_bps)
    while not env.done:
        if model is None:
            env.step(baseline_request(policy, env))
        else:
            import numpy as np
            observation = np.asarray(observation_vector(env.observation()), dtype=np.float32)
            action, _ = model.predict(observation, deterministic=True)
            env.step_action(int(np.asarray(action).item()))

    context = context if context is not None else episode_market_context(env.episode)
    thresholds = market_regimes or {}
    summary = asdict(env.summary())
    start = env.episode[0]["decision_time"]
    identifiers = {
        "episode_id": start.isoformat(), "episode_start": start.isoformat(),
        "day": start.date().isoformat(), "split": split, "policy": policy,
        "size": size, "parent_quantity_btc": float(quantity_btc), "fee_bps": float(fee_bps),
    }
    filled = summary["filled_btc"]
    execution_time = (
        fsum(step.filled_btc * step.step_index * STEP_SECONDS for step in env.history) / filled
        if filled > 0 else None
    )
    action_counts = Counter(int(step.action) for step in env.history if step.action is not None)
    trade_decisions = sum(step.filled_btc > 0 for step in env.history)
    record = {
        **identifiers, **summary, **context,
        "spread_regime": regime_label(context["arrival_spread_bps"], thresholds.get("arrival_spread_bps")),
        "volatility_regime": regime_label(context["episode_realized_volatility_bps"],
                                          thresholds.get("episode_realized_volatility_bps")),
        "average_execution_time_seconds": execution_time,
        "last_decision_seconds": env.history[-1].step_index * STEP_SECONDS,
        "decisions": len(env.history), "trade_decisions": trade_decisions,
        "mean_trade_size_btc": filled / trade_decisions if trade_decisions else None,
        "forced_terminal_liquidation": any(step.forced_terminal_liquidation for step in env.history),
        "final_decision_quantity_fraction": fsum(step.filled_btc for step in env.history
                                                 if step.step_index == 59) / quantity_btc,
        **{f"action_{action}_count": action_counts[action] for action in range(4)},
    }
    trace = []
    for step in env.history:
        entry = asdict(step)
        entry["decision_time"] = entry["decision_time"].isoformat()
        entry["action"] = int(step.action) if step.action is not None else None
        trace.append({**identifiers, "elapsed_seconds": step.step_index * STEP_SECONDS, **entry})
    return record, trace


def evaluate_episodes(episodes, sizes, *, models=None, fee_bps=10.0,
                      split="validation", market_regimes=None, max_episodes=None):
    """Stream an episode at a time, retaining detailed traces for the first one.

    The first chronological episode is fixed in advance, so the example chart
    cannot cherry-pick a favorable PPO outcome. Models map display names to
    already-loaded predict() objects; absent models still evaluates all baselines.
    """
    if not sizes or any(not isfinite(float(q)) or float(q) <= 0 for q in sizes.values()):
        raise ValueError("sizes must contain positive finite BTC quantities")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    models = models or {}
    if set(models).intersection(BASELINE_NAMES):
        raise ValueError("model names must not shadow baseline names")
    policies = [(name, None) for name in BASELINE_NAMES] + list(models.items())
    records, traces = [], []
    for episode_index, episode in enumerate(episodes):
        context = episode_market_context(episode)
        for size, quantity in sizes.items():
            for policy, model in policies:
                record, steps = evaluate_episode(
                    episode, policy, quantity, size=size, split=split, fee_bps=fee_bps,
                    model=model, market_regimes=market_regimes, context=context,
                )
                records.append(record)
                if episode_index == 0:
                    traces.extend(steps)
        if (episode_index + 1) % 50 == 0:
            print(f"Evaluated {episode_index + 1} {split} episodes across "
                  f"{len(sizes)} sizes and {len(policies)} policies.", flush=True)
        if max_episodes is not None and episode_index + 1 >= max_episodes:
            break
    if not records:
        raise ValueError("no complete episodes were available for evaluation")
    return records, traces


def write_csv(path, rows):
    """Write one consistent record schema; None is an empty CSV cell."""
    if not rows:
        raise ValueError("cannot write an empty result table")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_directory(path, *, overwrite=False):
    """Protect an existing run by default; overwrite never recursively deletes."""
    output = Path(path)
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not overwrite:
        raise FileExistsError(f"output is not empty: {output}; choose a new run or pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    return output


def installed_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "gymnasium", "stable-baselines3", "torch", "matplotlib"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def data_manifest(split, data_dir):
    """Hash only the processed files in the explicitly selected split."""
    manifest = []
    for day in SPLIT_DAYS[split]:
        path = Path(data_dir) / split / f"{day}_5s.csv.gz"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        manifest.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                         "sha256": digest.hexdigest()})
    return manifest


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("training", "validation", "testing"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                        help="repeat for saved PPO checkpoints; baseline policies are always included")
    parser.add_argument("--sizes", nargs="+", help="configured size names, e.g. small medium large")
    parser.add_argument("--fee-bps", type=float)
    parser.add_argument("--max-episodes", type=int, help="diagnostic subset, clearly recorded in metadata")
    parser.add_argument("--data-dir", type=Path, help="override config data_dir or the project data directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-figures", action="store_true", help="skip plotting for a software diagnostic")
    args = parser.parse_args(argv)
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    data_dir = args.data_dir or Path(config.get("data_dir", DATA_DIR))
    chosen = [name for item in args.sizes for name in item.split(",")] if args.sizes else list(config["sizes"])
    if len(set(chosen)) != len(chosen) or any(name not in config["sizes"] for name in chosen):
        parser.error("sizes must be unique names from the experiment configuration")
    sizes = {name: config["sizes"][name] for name in chosen}
    fee_bps = config["fee_bps"] if args.fee_bps is None else args.fee_bps
    if not isfinite(fee_bps) or not 0 <= fee_bps < 10_000:
        parser.error("fee-bps must be finite and between 0 inclusive and 10,000 exclusive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("max-episodes must be positive")

    model_paths = {}
    for specification in args.model:
        name, separator, path = specification.partition("=")
        if not separator or not name or not path or name in model_paths or name in BASELINE_NAMES:
            parser.error("each model must have a unique non-baseline NAME=PATH")
        checkpoint = Path(path)
        if not checkpoint.exists() and checkpoint.suffix != ".zip":
            checkpoint = Path(str(checkpoint) + ".zip")
        if not checkpoint.is_file():
            parser.error(f"checkpoint not found: {path}")
        model_paths[name] = checkpoint
    output = prepare_output_directory(args.output, overwrite=args.overwrite)
    versions = installed_versions()
    run_metadata = {
        "split": args.split, "fee_bps": fee_bps, "sizes": sizes,
        "data_dir": str(data_dir.resolve()),
        "primary_size": config.get("primary_size"), "deterministic_models": True,
        "max_episodes": args.max_episodes, "diagnostic_subset": args.max_episodes is not None,
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_regimes": config.get("market_regimes", {}),
        "data_manifest": data_manifest(args.split, data_dir),
        "models": {name: {"path": str(path.resolve()),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in model_paths.items()},
        "command": [sys.executable, *sys.argv],
    }
    _json(output / "config.json", config)
    _json(output / "versions.json", versions)
    _json(output / "run_metadata.json", run_metadata)
    models = {}
    if model_paths:
        import torch
        from stable_baselines3 import PPO
        # A tiny MLP at batch size one is faster and reproducible with one CPU thread.
        torch.set_num_threads(1)
        for name, path in model_paths.items():
            models[name] = PPO.load(str(path), device="cpu")
    stats = LoadStats()
    records, traces = evaluate_episodes(
        load_split(args.split, data_dir, stats=stats), sizes, models=models,
        fee_bps=fee_bps, split=args.split, market_regimes=config.get("market_regimes"),
        max_episodes=args.max_episodes,
    )
    pairs = paired_results(records)
    run_metadata.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        load_stats=asdict(stats), episode_count=len({r["episode_id"] for r in records}),
                        day_count=len({r["day"] for r in records}))
    summary = summarize_results(records, pairs, run_metadata)
    write_csv(output / "per_episode.csv", records)
    write_csv(output / "paired_results.csv", pairs)
    write_csv(output / "first_episode_steps.csv", traces)
    _json(output / "summary.json", summary)
    _json(output / "run_metadata.json", run_metadata)
    write_report(summary, output / "report.md")
    if not args.no_figures:
        make_figures(records, pairs, traces, output / "figures")
    print(f"Saved {len(records)} policy/episode/size results to {output.resolve()}")


if __name__ == "__main__":
    main()
