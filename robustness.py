"""Sensitivity analysis of SAVED execution results; no new policy or data replay.

Fee sensitivity reprices the same gross fills and quantities. Fees are absent
from the frozen agent's observations, so they cannot change its deterministic
actions. This does not estimate the behavior of a model retrained for other fees.
Hourly comparisons describe within-day heterogeneity, not independent trials or
out-of-sample statistical significance.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path
from statistics import mean

from reporting import distribution, paired_results, summarize_results


_STRING_FIELDS = {
    "episode_id", "episode_start", "day", "split", "policy", "size",
    "spread_regime", "volatility_regime",
}
_BOOLEAN_FIELDS = {"completed", "forced_terminal_liquidation"}
_PAIR_FIELDS = ("split", "episode_id", "size", "parent_quantity_btc", "fee_bps")
_BLOCK_FIELDS = ("split", "size", "parent_quantity_btc", "fee_bps", "benchmark")


def read_results(path):
    """Read evaluate.py's CSV with explicit booleans and missing-cost handling."""
    records = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("result CSV must have unique named columns")
        for line_number, raw in enumerate(reader, start=2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"result CSV row {line_number} has the wrong column count")
            row = {}
            for name, value in raw.items():
                if name in _STRING_FIELDS:
                    row[name] = value
                elif name in _BOOLEAN_FIELDS:
                    if value.lower() not in ("true", "false"):
                        raise ValueError(f"{name} must be True or False, at row {line_number}")
                    row[name] = value.lower() == "true"
                elif value == "":
                    row[name] = None
                else:
                    try:
                        number = float(value)
                    except ValueError as exc:
                        raise ValueError(f"{name} must be numeric, at row {line_number}") from exc
                    if not isfinite(number):
                        raise ValueError(f"{name} must be finite, at row {line_number}")
                    row[name] = number
            records.append(row)
    if not records:
        raise ValueError("result CSV must contain at least one episode")
    return records


def reprice_fees(row, fee_bps):
    """Recompute net accounting from actual gross proceeds; preserve all fills.

    For fee f in basis points, fee_USDT = gross_USDT * f / 10,000. Only completed
    orders have a full-order net implementation shortfall. Unsold inventory is
    separately marked; it never becomes cash or an invented execution.
    """
    fee = float(fee_bps)
    if not isfinite(fee) or not 0 <= fee < 10_000:
        raise ValueError("fee_bps must be finite in [0, 10000)")
    gross, filled = row["gross_received_usdt"], row["filled_btc"]
    quantity, arrival = row["parent_quantity_btc"], row["arrival_midprice"]
    residual_mark = row["residual_mark_usdt"]
    if any(not isfinite(value) for value in (gross, filled, quantity, arrival, residual_mark)):
        raise ValueError("source accounting must be finite")
    if gross < 0 or filled < 0 or residual_mark < 0 or quantity <= 0 or arrival <= 0:
        raise ValueError("source cash and fills must be nonnegative; Q and P0 must be positive")
    initial_notional = quantity * arrival
    if not isfinite(initial_notional) or initial_notional <= 0:
        raise ValueError("source parent arrival notional must be finite and positive")
    if not isinstance(row["completed"], bool):
        raise ValueError("completed must be a parsed boolean")
    if row["completed"] and filled <= 0:
        raise ValueError("a completed positive parent must contain actual fills")

    fees = gross * fee / 10_000.0
    net = gross - fees
    net_average = net / filled if filled > 0 else None
    net_shortfall = (10_000.0 * (1 - net_average / arrival)
                     if row["completed"] else None)
    score = 10_000.0 * (1 - net / initial_notional)
    return {
        **row, "source_fee_bps": row["fee_bps"], "fee_bps": fee,
        "fees_usdt": fees, "net_received_usdt": net,
        "net_average_fill_price": net_average,
        "net_implementation_shortfall_bps": net_shortfall,
        "marked_shortfall_bps": 10_000.0 * (1 - (net + residual_mark) / initial_notional),
        "completion_score_bps": score,
        # With gamma=1, the environment's undiscounted reward sums to -score.
        # This recomputes the algebraic total; it does not invent per-step logs.
        "total_reward": -score,
    }


def fee_sensitivity(records, fees):
    """Create one counterfactual row per recorded policy/episode and fee level."""
    if not records:
        raise ValueError("at least one recorded episode is required")
    fee_values = [float(value) for value in fees]
    if not fee_values or len(fee_values) != len(set(fee_values)):
        raise ValueError("stress fees must be a nonempty list of distinct values")
    if len({row["fee_bps"] for row in records}) != 1:
        raise ValueError("input must be one evaluation run at a single source fee")
    return [reprice_fees(row, fee) for fee in fee_values for row in records]


def hourly_comparisons(records, selected_policy, benchmarks=("twap", "immediate", "wait")):
    """Compare exact obligations, then aggregate their deltas by UTC start hour.

    Every hourly row reports its number of episodes and completed pairs. Hourly
    mean deltas are descriptive: neighboring hours and episodes are correlated.
    The analysis does not bootstrap or compute p-values/confidence intervals.
    """
    indexed = defaultdict(dict)
    for row in records:
        key = tuple(row[field] for field in _PAIR_FIELDS)
        if row["policy"] in indexed[key]:
            raise ValueError(f"duplicate saved policy result: {key} {row['policy']}")
        indexed[key][row["policy"]] = row
    blocks = defaultdict(list)
    found_selected = False
    for key, policies in indexed.items():
        if selected_policy not in policies:
            raise ValueError(f"selected policy {selected_policy!r} is missing for {key}")
        found_selected = True
        selected = policies[selected_policy]
        stamp = datetime.fromisoformat(selected["episode_start"].replace("Z", "+00:00"))
        if stamp.utcoffset() != timedelta(0):
            raise ValueError("episode start must be timezone-aware UTC")
        hour = stamp.replace(minute=0, second=0, microsecond=0).isoformat()
        for benchmark_name in benchmarks:
            if benchmark_name not in policies:
                raise ValueError(f"benchmark {benchmark_name!r} is missing for {key}")
            benchmark = policies[benchmark_name]
            completed = selected["completed"] and benchmark["completed"]
            if completed and (selected["net_implementation_shortfall_bps"] is None
                              or benchmark["net_implementation_shortfall_bps"] is None):
                raise ValueError("completed pairs must have net implementation shortfall")
            delta = (selected["net_implementation_shortfall_bps"]
                     - benchmark["net_implementation_shortfall_bps"] if completed else None)
            group = (selected["split"], selected["size"], selected["parent_quantity_btc"],
                     selected["fee_bps"], benchmark_name, hour)
            blocks[group].append({
                "net_shortfall_delta_bps": delta,
                "completion_score_delta_bps": selected["completion_score_bps"] - benchmark["completion_score_bps"],
                "selected_completed": selected["completed"],
                "benchmark_completed": benchmark["completed"],
            })
    if not found_selected:
        raise ValueError("no selected-policy results were available")

    hourly = []
    for key, rows in sorted(blocks.items()):
        costs = distribution(row["net_shortfall_delta_bps"] for row in rows)
        scores = distribution(row["completion_score_delta_bps"] for row in rows)
        hourly.append({
            **dict(zip(_BLOCK_FIELDS, key[:-1])), "selected_policy": selected_policy,
            "hour_utc": key[-1], "episodes": len(rows), "complete_pairs": costs["n"],
            "excluded_incomplete_pairs": len(rows) - costs["n"],
            "selected_completion_rate": mean(row["selected_completed"] for row in rows),
            "benchmark_completion_rate": mean(row["benchmark_completed"] for row in rows),
            "mean_net_shortfall_delta_bps": costs["mean"],
            "median_net_shortfall_delta_bps": costs["median"],
            "p05_net_shortfall_delta_bps": costs["p05"],
            "p95_net_shortfall_delta_bps": costs["p95"],
            "mean_completion_score_delta_bps": scores["mean"],
        })
    grouped = defaultdict(list)
    for row in hourly:
        grouped[tuple(row[field] for field in _BLOCK_FIELDS)].append(row)
    summaries = []
    for key, rows in grouped.items():
        available = [row["mean_net_shortfall_delta_bps"] for row in rows
                     if row["mean_net_shortfall_delta_bps"] is not None]
        summaries.append({
            **dict(zip(_BLOCK_FIELDS, key)), "selected_policy": selected_policy,
            "hours_observed": len(rows), "episodes": sum(row["episodes"] for row in rows),
            "hourly_mean_net_shortfall_delta_bps": distribution(available),
            "fraction_hours_with_lower_mean_cost": mean(value < -1e-9 for value in available) if available else None,
            "hourly_mean_completion_score_delta_bps": distribution(row["mean_completion_score_delta_bps"] for row in rows),
        })
    return hourly, summaries


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _plot_fee_sensitivity(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = summary["groups"]
    sizes = list(dict.fromkeys(row["size"] for row in groups))
    policies = list(dict.fromkeys(row["policy"] for row in groups))
    figure, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4), squeeze=False)
    for axis, size in zip(axes[0], sizes):
        for policy in policies:
            rows = sorted((row for row in groups if row["size"] == size and row["policy"] == policy),
                          key=lambda row: row["fee_bps"])
            axis.plot([row["fee_bps"] for row in rows],
                      [row["completion_score_bps"]["mean"] for row in rows],
                      marker="o", markersize=3, label=policy)
        axis.set(title=f"{size}: identical fills repriced", xlabel="Assumed fee (bps)",
                 ylabel="Mean completion score (bps; lower is better)")
        axis.legend(fontsize=7)
    figure.suptitle("Frozen policies and recorded quantities; no retraining or new execution", fontsize=11)
    figure.tight_layout()
    figure.savefig(output / "fee_sensitivity.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--selected-policy", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    source = args.results_dir / "per_episode.csv"
    records = read_results(source)
    config = json.loads(args.config.read_text())
    counterfactuals = fee_sensitivity(records, config["stress_fee_bps"])
    hourly, hourly_summary = hourly_comparisons(records, args.selected_policy)
    pairs = paired_results(counterfactuals)
    metadata = {
        "source_results": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "selected_policy": args.selected_policy,
        "source_fee_bps": records[0]["fee_bps"], "stress_fee_bps": config["stress_fee_bps"],
        "market_days": sorted({row["day"] for row in records}),
        "source_rows": len(records), "counterfactual_rows": len(counterfactuals),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Reprice the exact recorded gross proceeds/fills; all execution decisions and completion flags remain fixed.",
        "fee_scope": "Fees do not enter the frozen policy observation. Repricing does not simulate fee-aware retraining or account-specific historical exchange fees.",
        "hourly_scope": "Source-fee comparisons only. Hourly means are unweighted within-day descriptive slices, not iid replications, confidence intervals, or evidence of multi-day significance.",
    }
    summary = summarize_results(counterfactuals, pairs, metadata)
    summary["hourly_comparisons"] = hourly_summary
    output = args.output
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not args.overwrite:
        raise FileExistsError(f"output is not empty: {output}; choose a new directory or pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "fee_sensitivity.csv", counterfactuals)
    _write_csv(output / "fee_paired_results.csv", pairs)
    _write_csv(output / "hourly_comparisons.csv", hourly)
    _write_json(output / "summary.json", summary)
    _write_json(output / "config.json", config)
    if not args.no_figures:
        _plot_fee_sensitivity(summary, output)
    print(f"Repriced {len(records)} saved rows at {len(config['stress_fee_bps'])} fees; "
          f"wrote {len(hourly)} hourly comparisons to {output.resolve()}")


if __name__ == "__main__":
    main()
