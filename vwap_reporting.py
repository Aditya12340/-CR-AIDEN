"""Additive VWAP benchmark reporting; the frozen original study is unchanged.

An executable VWAP schedule uses a volume forecast fitted on training data.
Realized market VWAP is a separate, ex-post reference price. It is never a fill
price, a policy observation, or knowledge available to the schedule at arrival.
"""

from collections import defaultdict
from math import isclose, isfinite
from pathlib import Path
from statistics import mean

from reporting import distribution


_PAIR_FIELDS = ("split", "episode_id", "size", "parent_quantity_btc", "fee_bps")
_GROUP_FIELDS = ("split", "size", "parent_quantity_btc", "fee_bps", "policy")
_TIE_BPS = 1e-9


def add_market_metrics(row, market):
    """Attach completed-order sell slippage versus observed market VWAP.

    Positive slippage means selling below the reference. Net price includes
    actual execution fees; the market reference itself is the traded price,
    without a hypothetical fee deduction. No-volume or incomplete orders have
    no VWAP score, while their original completion score remains available.
    """
    volume = float(market["market_volume_btc"])
    notional = float(market["market_notional_usdt"])
    price = market["market_vwap_usdt_per_btc"]
    if not all(isfinite(value) and value >= 0 for value in (volume, notional)):
        raise ValueError("market volume and notional must be finite and nonnegative")
    if volume == 0:
        if notional != 0 or price is not None:
            raise ValueError("zero market volume requires zero notional and no VWAP")
    else:
        if price is None or not isfinite(float(price)) or float(price) <= 0:
            raise ValueError("positive market volume requires a finite positive VWAP")
        price = float(price)
        if not isclose(price, notional / volume, rel_tol=1e-10):
            raise ValueError("market VWAP must equal notional divided by volume")
    if not isinstance(row["completed"], bool):
        raise ValueError("completed must be a parsed boolean")
    gross_slippage = net_slippage = None
    if row["completed"] and price is not None:
        gross_price, net_price = row["average_fill_price"], row["net_average_fill_price"]
        if any(value is None or not isfinite(value) or value < 0 for value in (gross_price, net_price)):
            raise ValueError("completed orders require finite nonnegative fill prices")
        gross_slippage = 10_000.0 * (1 - gross_price / price)
        net_slippage = 10_000.0 * (1 - net_price / price)
    return {
        **row, "market_vwap_usdt_per_btc": price,
        "market_volume_btc": volume, "market_notional_usdt": notional,
        "market_bins": market["bins"],
        "gross_vwap_slippage_bps": gross_slippage,
        "net_vwap_slippage_bps": net_slippage,
    }


def compare_and_summarize(records, selected_policy, metadata):
    """Pair the frozen selected policy against executable VWAP and TWAP.

    Exact obligation keys include split, episode, parent size, and fees. Both
    orders must complete for execution-cost wins. The cash completion score
    includes every matched order so incomplete strategies cannot disappear.
    """
    if not records:
        raise ValueError("VWAP comparison requires actual episode results")
    indexed, groups = defaultdict(dict), defaultdict(list)
    for row in records:
        key = tuple(row[field] for field in _PAIR_FIELDS)
        if row["policy"] in indexed[key]:
            raise ValueError(f"duplicate policy/obligation: {key} {row['policy']}")
        indexed[key][row["policy"]] = row
        groups[tuple(row[field] for field in _GROUP_FIELDS)].append(row)
    pairs = []
    for key, policies in indexed.items():
        missing = {selected_policy, "vwap", "twap"}.difference(policies)
        if missing:
            raise ValueError(f"unmatched obligation {key}: missing {sorted(missing)}")
        selected = policies[selected_policy]
        for benchmark in ("vwap", "twap"):
            reference = policies[benchmark]
            # Ex-post reference must be identical across policies on one tape.
            for field in ("market_vwap_usdt_per_btc", "market_volume_btc", "market_notional_usdt", "market_bins"):
                if selected[field] != reference[field]:
                    raise ValueError(f"policies have inconsistent market reference: {field}")
            both = selected["completed"] and reference["completed"]
            arrival_delta = vwap_delta = None
            if both:
                costs = [item["net_implementation_shortfall_bps"] for item in (selected, reference)]
                if any(value is None for value in costs):
                    raise ValueError("completed orders require net arrival shortfall")
                arrival_delta = costs[0] - costs[1]
                if selected["market_vwap_usdt_per_btc"] is not None:
                    vwap_delta = selected["net_vwap_slippage_bps"] - reference["net_vwap_slippage_bps"]
            pairs.append({
                **dict(zip(_PAIR_FIELDS, key)), "policy": selected_policy, "benchmark": benchmark,
                "both_completed": bool(both), "market_vwap_available": selected["market_vwap_usdt_per_btc"] is not None,
                "arrival_shortfall_delta_bps": arrival_delta,
                "net_vwap_slippage_delta_bps": vwap_delta,
                "completion_score_delta_bps": selected["completion_score_bps"] - reference["completion_score_bps"],
                "arrival_cost_win": arrival_delta < -_TIE_BPS if arrival_delta is not None else None,
                "arrival_cost_tie": abs(arrival_delta) <= _TIE_BPS if arrival_delta is not None else None,
                "vwap_cost_win": vwap_delta < -_TIE_BPS if vwap_delta is not None else None,
            })

    group_summaries = []
    for key, rows in groups.items():
        group_summaries.append({
            **dict(zip(_GROUP_FIELDS, key)), "episodes": len(rows),
            "completed_episodes": sum(row["completed"] for row in rows),
            "completion_rate": mean(row["completed"] for row in rows),
            "zero_market_volume_episodes": sum(row["market_volume_btc"] == 0 for row in rows),
            "completion_fraction": distribution(row["completion_fraction"] for row in rows),
            "net_arrival_shortfall_bps": distribution(row["net_implementation_shortfall_bps"] for row in rows),
            "completion_score_bps": distribution(row["completion_score_bps"] for row in rows),
            "gross_vwap_slippage_bps": distribution(row["gross_vwap_slippage_bps"] for row in rows),
            "net_vwap_slippage_bps": distribution(row["net_vwap_slippage_bps"] for row in rows),
        })
    compared = defaultdict(list)
    for row in pairs:
        compared[(*tuple(row[field] for field in _GROUP_FIELDS), row["benchmark"])].append(row)
    comparisons = []
    for key, rows in compared.items():
        completed = [row for row in rows if row["both_completed"]]
        market_pairs = [row for row in completed if row["market_vwap_available"]]
        comparisons.append({
            **dict(zip(_GROUP_FIELDS, key[:-1])), "benchmark": key[-1],
            "episodes": len(rows), "complete_pairs": len(completed),
            "excluded_incomplete_pairs": len(rows) - len(completed), "market_vwap_pairs": len(market_pairs),
            "arrival_shortfall_delta_bps": distribution(row["arrival_shortfall_delta_bps"] for row in rows),
            "net_vwap_slippage_delta_bps": distribution(row["net_vwap_slippage_delta_bps"] for row in rows),
            "completion_score_delta_bps": distribution(row["completion_score_delta_bps"] for row in rows),
            "arrival_cost_win_rate": mean(row["arrival_cost_win"] for row in completed) if completed else None,
            "arrival_cost_tie_rate": mean(row["arrival_cost_tie"] for row in completed) if completed else None,
            "vwap_cost_win_rate": mean(row["vwap_cost_win"] for row in market_pairs) if market_pairs else None,
        })
    primary = metadata.get("primary_size", metadata.get("config", {}).get("primary_size", "medium"))
    summary = {
        "metadata": metadata, "selected_policy": selected_policy,
        "groups": group_summaries, "benchmark_comparisons": comparisons,
        "primary_test": {
            "size": primary,
            "groups": [group for group in group_summaries if group["split"] == "testing" and group["size"] == primary],
            "comparisons": [group for group in comparisons if group["split"] == "testing" and group["size"] == primary],
        },
        "limitations": [
            "This is a post-hoc benchmark extension to the original frozen study. No PPO checkpoint is retrained or selected again; it is not a new untouched holdout claim.",
            "Executable VWAP uses a training-only forecast: mean backward five-second trade volume by UTC minute, giving 60 decision-clock weights. It does not know this episode's future traded volume.",
            "Realized market VWAP uses 59 observed trade bins from rows 1 through 59: (arrival, last decision], a 295-second interval. The pre-arrival bin in row 0 is excluded; the final five seconds to the nominal deadline are unobserved.",
            "Arrival shortfall and realized-VWAP slippage answer different questions. Positive values are worse for a sell; the market VWAP is a benchmark, never a guaranteed executable price.",
            "Completed-order metrics exclude incomplete obligations; all-order completion scores and completion rates must be read alongside them. Zero-volume market windows have no realized VWAP metric.",
            "Validation and testing each cover one market day. Quantiles and win fractions are descriptive; correlated five-minute episodes do not establish multi-day statistical significance.",
            "Historical replay does not model own market impact, passive queues, latency, or historical exchange lot filters. The training volume profile comes from only five separate days.",
        ],
    }
    return pairs, summary


def write_report(summary, output_path):
    """Write actual primary-test numbers followed by all splits and sizes."""
    def fmt(value, places=3):
        return "n/a" if value is None else f"{value:.{places}f}"

    lines = ["# VWAP benchmark extension", "",
             f"Frozen selected policy: **{summary['selected_policy']}**. "
             "This additive study compares it with a feasible volume-forecast schedule and TWAP.", "",
             "The executable VWAP policy uses training-only volume weights. Realized market VWAP is a separate ex-post reference computed from the observed 59 post-arrival trade bins.", "",
             "## Primary test comparison", "",
             f"Order-size regime: **{summary['primary_test']['size']}**. Costs are in basis points; lower is better.", "",
             "| Policy | Episodes | Complete % | Net arrival IS mean | Arrival IS P95 | Worst 5% IS | Completion score mean |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for group in summary["primary_test"]["groups"]:
        cost = group["net_arrival_shortfall_bps"]
        lines.append(f"| {group['policy']} | {group['episodes']} | {100 * group['completion_rate']:.1f} "
                     f"| {fmt(cost['mean'])} | {fmt(cost['p95'])} | {fmt(cost['worst_5pct_mean'])} "
                     f"| {fmt(group['completion_score_bps']['mean'])} |")
    lines.extend(["", "Market-VWAP slippage below uses the same observed market reference. "
                  "Gross excludes fees; net deducts fees from actual execution proceeds.", "",
                  "| Policy | Eligible completed episodes | Gross VWAP slippage mean | Net VWAP slippage mean |",
                  "|---|---:|---:|---:|"])
    for group in summary["primary_test"]["groups"]:
        gross, net = group["gross_vwap_slippage_bps"], group["net_vwap_slippage_bps"]
        lines.append(f"| {group['policy']} | {net['n']} | {fmt(gross['mean'])} | {fmt(net['mean'])} |")
    lines.extend(["", "Paired differences below are **selected PPO minus the named schedule**, so negative favors PPO. "
                  "Execution-cost wins require both obligations to complete.", "",
                  "| Benchmark | Complete pairs | Mean arrival-cost Δ | P95 Δ | Arrival-cost win % | Mean market-VWAP Δ | Mean completion-score Δ |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for group in summary["primary_test"]["comparisons"]:
        win = group["arrival_cost_win_rate"]
        delta = group["arrival_shortfall_delta_bps"]
        lines.append(f"| {group['benchmark']} | {group['complete_pairs']} | {fmt(delta['mean'])} "
                     f"| {fmt(delta['p95'])} | {fmt(100 * win if win is not None else None, 1)} "
                     f"| {fmt(group['net_vwap_slippage_delta_bps']['mean'])} "
                     f"| {fmt(group['completion_score_delta_bps']['mean'])} |")
    lines.extend(["", "## All splits and sizes", "",
                  "| Split | Size (BTC) | Policy | Complete % | Net arrival IS mean | P95 | Worst 5% | Net market-VWAP slippage | Completion score | No-market-volume episodes |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"])
    for group in summary["groups"]:
        cost = group["net_arrival_shortfall_bps"]
        lines.append(f"| {group['split']} | {group['size']} ({group['parent_quantity_btc']:.6g}) "
                     f"| {group['policy']} | {100 * group['completion_rate']:.1f} "
                     f"| {fmt(cost['mean'])} | {fmt(cost['p95'])} | {fmt(cost['worst_5pct_mean'])} "
                     f"| {fmt(group['net_vwap_slippage_bps']['mean'])} | {fmt(group['completion_score_bps']['mean'])} "
                     f"| {group['zero_market_volume_episodes']} |")
    lines.extend(["", "## Figures", "",
                  "![Arrival-cost and realized-market-VWAP distributions](figures/vwap_cost_distributions.png)", "",
                  "![Paired cost differences and completion rates](figures/vwap_paired_and_completion.png)", "",
                  "## Reading the evidence", ""])
    lines.extend(f"- {note}" for note in summary["limitations"])
    lines.extend(["", "`summary.json` retains full distributions, excluded-pair counts, and source metadata; "
                  "the accompanying per-episode and paired CSVs retain the observations behind these tables."])
    Path(output_path).write_text("\n".join(lines) + "\n")


def make_figures(records, summary, output_dir):
    """Plot held-out distributions and paired effects without hiding completion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    available = {row["split"] for row in records}
    split = next(name for name in ("testing", "validation", "training") if name in available)
    rows = [row for row in records if row["split"] == split]
    sizes = list(dict.fromkeys(row["size"] for row in rows))
    policies = ["twap", "vwap", summary["selected_policy"]]
    colors = dict(zip(policies, ("#5a6978", "#cc7b17", "#186d8b")))
    fig, axes = plt.subplots(2, len(sizes), figsize=(5 * len(sizes), 7), squeeze=False)
    for column, size in enumerate(sizes):
        for index, (field, label) in enumerate((
            ("net_implementation_shortfall_bps", "Net arrival shortfall (bps)"),
            ("net_vwap_slippage_bps", "Net slippage vs realized market VWAP (bps)"),
        )):
            axis = axes[index, column]
            for policy in policies:
                values = sorted(row[field] for row in rows if row["size"] == size
                                and row["policy"] == policy and row[field] is not None)
                if values:
                    axis.step(values, [(i + 1) / len(values) for i in range(len(values))],
                              where="post", label=policy, color=colors[policy])
            axis.set(title=f"{split}: {size}", xlabel=label, ylabel="Completed-order fraction", ylim=(0, 1.02))
            axis.legend(fontsize=8)
    fig.suptitle("VWAP extension: completed-order cost distributions; lower cost is better", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "vwap_cost_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for j, benchmark in enumerate(("vwap", "twap")):
        for i, size in enumerate(sizes):
            group = next(group for group in summary["benchmark_comparisons"]
                         if group["split"] == split and group["size"] == size and group["benchmark"] == benchmark)
            stats = group["arrival_shortfall_delta_bps"]
            if stats["n"]:
                center = stats["median"]
                axes[0].errorbar(i + (j - 0.5) * 0.22, center,
                                 yerr=[[center - stats["p05"]], [stats["p95"] - center]],
                                 fmt="o", capsize=4, color=colors[benchmark],
                                 label=f"vs {benchmark}" if i == 0 else None)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(xticks=range(len(sizes)), xticklabels=sizes,
                ylabel="Selected PPO minus benchmark (bps)",
                title="Paired median and descriptive 5th–95th percentiles")
    axes[0].legend()
    for j, policy in enumerate(policies):
        completion = [next(group["completion_rate"] for group in summary["groups"]
                           if group["split"] == split and group["size"] == size and group["policy"] == policy)
                      for size in sizes]
        axes[1].bar([i + (j - 1) * 0.24 for i in range(len(sizes))], completion,
                    width=0.23, label=policy, color=colors[policy])
    axes[1].set(xticks=range(len(sizes)), xticklabels=sizes, ylim=(0, 1.05),
                ylabel="Fraction of orders completed", title="All obligations: completion by parent size")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{split}: negative paired cost favors selected PPO; bands are not confidence intervals", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "vwap_paired_and_completion.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
