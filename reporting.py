"""Paired execution comparisons, descriptive statistics, and exportable figures.

Lower implementation shortfall and lower completion score are better. A cost
comparison excludes a pair unless BOTH policies completed; the separate cash
completion score retains every episode, including failures to finish. No test
of statistical significance treats intraday windows as independent market days.
"""

from collections import defaultdict
from math import ceil, fsum, isfinite, sqrt
from pathlib import Path
from statistics import mean, median


PAIR_FIELDS = ("split", "episode_id", "size", "parent_quantity_btc", "fee_bps")
GROUP_FIELDS = ("split", "policy", "size", "parent_quantity_btc", "fee_bps")
_TIE_TOLERANCE_BPS = 1e-9


def _key(row, fields):
    return tuple(row[field] for field in fields)


def distribution(values):
    """Linear-interpolated quantiles and worst ceil(5% * n) observations.

    The worst tail is the largest cost, so a positive tail means costly sells.
    This tail mean is descriptive, not an estimated population risk guarantee.
    Missing metrics (incomplete-order IS) are excluded and counted through n.
    """
    numbers = sorted(float(value) for value in values if value is not None)
    if any(not isfinite(value) for value in numbers):
        raise ValueError("reported metrics must be finite or None")
    if not numbers:
        return {name: (0 if name == "n" else None) for name in (
            "n", "mean", "median", "std", "p05", "p95", "min", "max", "worst_5pct_mean"
        )}

    def quantile(probability):
        position = (len(numbers) - 1) * probability
        left = int(position)
        right = min(left + 1, len(numbers) - 1)
        return numbers[left] + (position - left) * (numbers[right] - numbers[left])

    average = mean(numbers)
    deviation = (sqrt(fsum((value - average) ** 2 for value in numbers)
                      / (len(numbers) - 1)) if len(numbers) > 1 else None)
    return {
        "n": len(numbers), "mean": average, "median": median(numbers),
        "std": deviation, "p05": quantile(0.05), "p95": quantile(0.95),
        "min": numbers[0], "max": numbers[-1],
        "worst_5pct_mean": mean(numbers[-max(1, ceil(0.05 * len(numbers))):]),
    }


def paired_results(records):
    """Join each policy to TWAP on the exact episode, quantity, size, and fee.

    Positive policy-minus-TWAP cost deltas mean the policy did worse. Net cash
    deltas have the opposite interpretation: positive means more USDT received.
    Missing benchmark pairs and duplicate results are errors, not dropped rows.
    """
    by_pair = defaultdict(dict)
    for row in records:
        key = _key(row, PAIR_FIELDS)
        if row["policy"] in by_pair[key]:
            raise ValueError(f"duplicate policy result for {key}: {row['policy']}")
        by_pair[key][row["policy"]] = row
    pairs = []
    for key, policies in by_pair.items():
        if "twap" not in policies:
            raise ValueError(f"missing matched TWAP result for {key}")
        benchmark = policies["twap"]
        for policy, row in policies.items():
            both_completed = bool(row["completed"] and benchmark["completed"])
            delta = None
            if both_completed:
                if any(item["net_implementation_shortfall_bps"] is None
                       for item in (row, benchmark)):
                    raise ValueError("completed orders must have net implementation shortfall")
                delta = (row["net_implementation_shortfall_bps"]
                         - benchmark["net_implementation_shortfall_bps"])
            pairs.append({
                **dict(zip(PAIR_FIELDS, key)), "policy": policy,
                "day": row["day"],
                "both_completed": both_completed,
                "net_shortfall_delta_bps": delta,
                "completion_score_delta_bps": (row["completion_score_bps"]
                                                - benchmark["completion_score_bps"]),
                "net_proceeds_delta_usdt": (row["net_received_usdt"]
                                            - benchmark["net_received_usdt"]),
                "completion_fraction_delta": (row["completion_fraction"]
                                               - benchmark["completion_fraction"]),
                "cost_win": delta < -_TIE_TOLERANCE_BPS if delta is not None else None,
                "cost_tie": abs(delta) <= _TIE_TOLERANCE_BPS if delta is not None else None,
                "spread_regime": row["spread_regime"],
                "volatility_regime": row["volatility_regime"],
            })
    return pairs


def _pair_summary(rows):
    eligible = [row for row in rows if row["both_completed"]]
    return {
        "paired_episodes": len(rows), "complete_pairs": len(eligible),
        "excluded_incomplete_pairs": len(rows) - len(eligible),
        "cost_win_rate": mean(row["cost_win"] for row in eligible) if eligible else None,
        "cost_tie_rate": mean(row["cost_tie"] for row in eligible) if eligible else None,
        "net_shortfall_delta_bps": distribution(row["net_shortfall_delta_bps"] for row in rows),
        "completion_score_delta_bps": distribution(row["completion_score_delta_bps"] for row in rows),
        "net_proceeds_delta_usdt": distribution(row["net_proceeds_delta_usdt"] for row in rows),
    }


def summarize_results(records, pairs, metadata=None):
    """Describe policy/size groups and fixed training-threshold market regimes."""
    if not records:
        raise ValueError("at least one episode result is required")
    grouped, paired = defaultdict(list), defaultdict(list)
    for row in records:
        grouped[_key(row, GROUP_FIELDS)].append(row)
    for row in pairs:
        paired[_key(row, GROUP_FIELDS)].append(row)
    summaries = []
    for key, rows in grouped.items():
        summaries.append({
            **dict(zip(GROUP_FIELDS, key)), "episodes": len(rows),
            "completed_episodes": sum(row["completed"] for row in rows),
            "completion_rate": mean(row["completed"] for row in rows),
            "completion_fraction": distribution(row["completion_fraction"] for row in rows),
            "net_shortfall_bps": distribution(row["net_implementation_shortfall_bps"] for row in rows),
            "gross_shortfall_bps": distribution(row["implementation_shortfall_bps"] for row in rows),
            "completion_score_bps": distribution(row["completion_score_bps"] for row in rows),
            "marked_shortfall_bps": distribution(row["marked_shortfall_bps"] for row in rows),
            "net_received_usdt": distribution(row["net_received_usdt"] for row in rows),
            "total_reward": distribution(row["total_reward"] for row in rows),
            "average_execution_time_seconds": distribution(row["average_execution_time_seconds"] for row in rows),
            "trade_decisions": distribution(row["trade_decisions"] for row in rows),
            "action_counts": {str(action): sum(row[f"action_{action}_count"] for row in rows)
                              for action in range(4)},
            "forced_terminal_episode_fraction": mean(row["forced_terminal_liquidation"] for row in rows),
            "final_decision_quantity_fraction": distribution(row["final_decision_quantity_fraction"] for row in rows),
            "paired_vs_twap": _pair_summary(paired[key]),
        })

    regimes = []
    for field in ("spread_regime", "volatility_regime"):
        buckets = defaultdict(list)
        for row in pairs:
            buckets[(*_key(row, GROUP_FIELDS), row[field])].append(row)
        for key, rows in buckets.items():
            regimes.append({**dict(zip(GROUP_FIELDS, key[:-1])),
                            "regime_feature": field, "regime": key[-1],
                            **_pair_summary(rows)})
    by_day, paired_days = defaultdict(list), defaultdict(list)
    for row in records:
        by_day[(*_key(row, GROUP_FIELDS), row["day"])].append(row)
    for row in pairs:
        paired_days[(*_key(row, GROUP_FIELDS), row["day"])].append(row)
    day_groups = []
    for key, rows in by_day.items():
        day_groups.append({
            **dict(zip(GROUP_FIELDS, key[:-1])), "day": key[-1], "episodes": len(rows),
            "completion_rate": mean(row["completed"] for row in rows),
            "net_shortfall_bps": distribution(row["net_implementation_shortfall_bps"] for row in rows),
            "completion_score_bps": distribution(row["completion_score_bps"] for row in rows),
            "paired_vs_twap": _pair_summary(paired_days[key]),
        })
    return {
        "metadata": metadata or {}, "groups": summaries, "regime_groups": regimes,
        "day_groups": day_groups,
        "limitations": [
            "Validation and testing each contain one market day. Intraday episodes are correlated; no iid confidence intervals or significance claims are reported.",
            "Cost deltas compare identical episodes, parent quantities, and fees. Completed-order shortfall excludes incomplete pairs; completion scores retain all episodes.",
            "Episode realized volatility uses the full episode only for descriptive ex-post stratification; it is never an agent input or sizing signal.",
            "Regime cutoffs are fixed from training data, not estimated again on validation or testing.",
            "Historical replay models displayed depth and fees, but no own market impact, queue position, exchange lot filters, or execution latency.",
            "A cash-only completion score penalizes unsold BTC heavily. It is a project objective, not the economic value of retained inventory.",
        ],
    }


def write_report(summary, path):
    """Write a short human-readable entry point to the detailed CSV/JSON files."""
    def number(value, digits=3):
        return "n/a" if value is None else f"{value:.{digits}f}"

    metadata = summary["metadata"]
    lines = [
        "# Paired execution evaluation", "",
        f"Split: **{metadata.get('split', 'unspecified')}**. "
        f"Fee: **{metadata.get('fee_bps', 'unspecified')} bps** of executed notional.", "",
        "Costs and deltas are in basis points; lower is better. "
        "A negative policy-minus-TWAP delta favors the policy. "
        "Shortfall statistics below include only completed orders; paired deltas require both orders to complete.", "",
        "| Policy | Size (BTC) | Episodes | Complete % | Net IS mean | Median | P95 | Worst 5% mean | Mean Δ vs TWAP | Win % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary["groups"]:
        cost, pair = group["net_shortfall_bps"], group["paired_vs_twap"]
        win = pair["cost_win_rate"]
        lines.append(
            f"| {group['policy']} | {group['size']} ({group['parent_quantity_btc']:.6g}) "
            f"| {group['episodes']} | {100 * group['completion_rate']:.1f} "
            f"| {number(cost['mean'])} | {number(cost['median'])} | {number(cost['p95'])} "
            f"| {number(cost['worst_5pct_mean'])} | {number(pair['net_shortfall_delta_bps']['mean'])} "
            f"| {number(100 * win if win is not None else None, 1)} |"
        )
    lines.extend(["", "## Completion and behavior", "",
                  "The completion score retains incomplete episodes; zero cash proceeds gives a 10,000-bp score. "
                  "Execution time is filled-BTC-weighted seconds from arrival. Forced means the last action was overridden to request the remainder.", "",
                  "| Policy | Size | Mean completion score | Mean execution time (s) | Forced terminal % | Excluded cost pairs |",
                  "|---|---|---:|---:|---:|---:|"])
    for group in summary["groups"]:
        lines.append(f"| {group['policy']} | {group['size']} "
                     f"| {number(group['completion_score_bps']['mean'])} "
                     f"| {number(group['average_execution_time_seconds']['mean'], 1)} "
                     f"| {100 * group['forced_terminal_episode_fraction']:.1f} "
                     f"| {group['paired_vs_twap']['excluded_incomplete_pairs']} |")
    lines.extend(["", "## Files", "",
                  "- `per_episode.csv`: complete accounting for every policy/episode/size.",
                  "- `paired_results.csv`: exact matches against TWAP, with explicit incomplete exclusions.",
                  "- `first_episode_steps.csv`: actual fills/actions for the fixed first episode, not a selected winner.",
                  "- `summary.json`: quantiles, tails, completion, behavior, and training-threshold regime breakdowns.",
                  "- `figures/`: standalone PNG charts; `config.json` and `versions.json` record reproduction details.",
                  "", "## Interpretation limits", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    Path(path).write_text("\n".join(lines) + "\n")


def make_figures(records, pairs, traces, output_dir):
    """Save four small scientific figures from actual results; never resample test data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sizes = list(dict.fromkeys(row["size"] for row in records))
    policies = list(dict.fromkeys(row["policy"] for row in records))
    colors = dict(zip(policies, plt.get_cmap("tab10").colors * (1 + len(policies) // 10)))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4), squeeze=False)
    for axis, size in zip(axes[0], sizes):
        for policy in policies:
            values = sorted(row["net_implementation_shortfall_bps"] for row in records
                            if row["size"] == size and row["policy"] == policy
                            and row["completed"])
            if values:
                axis.step(values, [(i + 1) / len(values) for i in range(len(values))],
                          where="post", label=policy, color=colors[policy])
        axis.set(title=f"{size}: completed-order cost", xlabel="Net shortfall (bps; lower is better)",
                 ylabel="Fraction of completed episodes", ylim=(0, 1.02))
        axis.legend(fontsize=7)
    finish(fig, "cost_distribution.png")

    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4), squeeze=False)
    for axis, size in zip(axes[0], sizes):
        for policy in policies:
            rows = [row for row in records if row["size"] == size and row["policy"] == policy]
            # Executed-only slippage can look good with low completion. Showing
            # every episode exposes that trade-off; unexecuted orders use x=0
            # only as a plotted missing-value marker, never in cost statistics.
            executed = [row for row in rows if row["net_average_fill_price"] is not None]
            axis.scatter([10_000 * (1 - row["net_average_fill_price"] / row["arrival_midprice"])
                          for row in executed],
                         [row["completion_fraction"] for row in executed],
                         s=12, alpha=0.35, label=policy, color=colors[policy])
            no_fills = [row for row in rows if row["net_average_fill_price"] is None]
            if no_fills:
                axis.scatter([0] * len(no_fills), [0] * len(no_fills), marker="x", color=colors[policy])
        axis.set(title=f"{size}: completion versus executed cost", ylim=(-0.03, 1.04),
                 xlabel="Executed-only net slippage (bps)", ylabel="Parent quantity completed")
        axis.legend(fontsize=7)
    fig.suptitle("Executed-only cost is not a full-order score; × at (0, 0) denotes no fills", y=1.02)
    finish(fig, "completion_vs_cost.png")

    fig, axes = plt.subplots(len(sizes), 3, figsize=(14, 3.5 * len(sizes)), squeeze=False)
    for row_axes, size in zip(axes, sizes):
        for axis, field in zip(row_axes, (None, "spread_regime", "volatility_regime")):
            buckets = ["all"] if field is None else ["low", "medium", "high"]
            compared = [policy for policy in policies if policy != "twap"]
            width = 0.8 / max(1, len(compared))
            for j, policy in enumerate(compared):
                centers, centers_mean, lower, upper = [], [], [], []
                for i, bucket in enumerate(buckets):
                    values = [pair["net_shortfall_delta_bps"] for pair in pairs
                              if pair["size"] == size and pair["policy"] == policy
                              and pair["both_completed"]
                              and (field is None or pair[field] == bucket)]
                    stats = distribution(values)
                    if stats["n"]:
                        center = stats["median"]
                        centers.append(i - 0.4 + width * (j + 0.5))
                        centers_mean.append(center)
                        lower.append(center - stats["p05"])
                        upper.append(stats["p95"] - center)
                axis.errorbar(centers, centers_mean, yerr=[lower, upper], fmt="o", capsize=3,
                              color=colors[policy], label=policy)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set(xticks=range(len(buckets)), xticklabels=buckets,
                     title=f"{size}: {field or 'all episodes'}",
                     ylabel="Cost Δ vs TWAP (bps)")
            axis.legend(fontsize=7)
    fig.suptitle("Paired median and descriptive 5th–95th percentiles; negative favors policy", y=1.01)
    finish(fig, "paired_deltas_by_regime.png")

    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4), squeeze=False)
    for axis, size in zip(axes[0], sizes):
        for policy in policies:
            rows = [row for row in traces if row["size"] == size and row["policy"] == policy]
            if rows:
                times = [0] + [row["elapsed_seconds"] for row in rows] + [300]
                inventory = [1] + [row["remaining_btc"] / row["parent_quantity_btc"] for row in rows]
                inventory.append(inventory[-1])
                axis.step(times, inventory, where="post", label=policy, color=colors[policy])
        axis.set(title=f"{size}: fixed first episode", xlabel="Seconds from arrival",
                 ylabel="Inventory remaining / Q", xlim=(0, 300), ylim=(-0.03, 1.03))
        axis.legend(fontsize=7)
    finish(fig, "first_episode_inventory.png")
