"""Summarize measured runs and export learning curves, without new evaluation."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def group(summary, policy, size):
    return next(row for row in summary["groups"]
                if row["policy"] == policy and row["size"] == size)


def number(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def learning_figures(run, manifests, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for manifest in manifests:
        seed = manifest["seed"]
        with (run / "ppo" / f"seed_{seed}" / "validation.csv").open() as handle:
            validation = list(csv.DictReader(handle))
        axes[0].plot([int(row["num_timesteps"]) for row in validation],
                     [float(row["mean_completion_score_bps"]) for row in validation],
                     marker="o", label=f"seed {seed}")
        with (run / "ppo" / f"seed_{seed}" / "progress.csv").open() as handle:
            progress = list(csv.DictReader(handle))
        measured = [row for row in progress if row.get("time/total_timesteps")
                    and row.get("rollout/ep_rew_mean")]
        axes[1].plot([float(row["time/total_timesteps"]) for row in measured],
                     [float(row["rollout/ep_rew_mean"]) for row in measured],
                     label=f"seed {seed}", alpha=.8)
    subset_size = len(manifests[0]["validation_episode_indices"])
    axes[0].set(title=f"Fixed {subset_size}-episode validation subset", xlabel="Training interactions",
                ylabel="Mean completion score (bps; lower better)")
    axes[1].set(title="Training: rolling mean of last 100 episodes",
                xlabel="Training interactions", ylabel="Undiscounted reward (bps)")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)


def build_report(run):
    run = Path(run)
    frozen = read_json(run / "frozen_selection.json")
    config = frozen["config"]
    summaries = {name: read_json(run / name / "summary.json")
                 for name in ("training", "validation", "testing")}
    robustness = read_json(run / "robustness" / "summary.json")
    test = summaries["testing"]
    metadata = test["metadata"]
    selected = frozen["selected"]["policy"]
    primary = config["primary_size"]
    # The published result must use the validation-chosen file, after freezing.
    if metadata["config_sha256"] != frozen["config_sha256"]:
        raise ValueError("test configuration differs from the frozen experiment")
    if datetime.fromisoformat(metadata["started_at_utc"]) <= datetime.fromisoformat(frozen["frozen_at_utc"]):
        raise ValueError("held-out evaluation must start after selection was frozen")
    for name, checkpoint in frozen["checkpoints"].items():
        if metadata["models"][name]["sha256"] != checkpoint["sha256"]:
            raise ValueError("test checkpoint differs from the frozen model")
    manifests = [read_json(run / "ppo" / f"seed_{seed}" / "manifest.json")
                 for seed in config["seeds"]]
    subset_size = len(manifests[0]["validation_episode_indices"])
    training_meta = summaries["training"]["metadata"]
    output = run / "figures"
    output.mkdir(exist_ok=True)
    learning_figures(run, manifests, output)

    total_steps = sum(item["actual_timesteps"] for item in manifests)
    selected_test = group(test, selected, primary)
    twap_test = group(test, "twap", primary)
    difference = (selected_test["completion_score_bps"]["mean"]
                  - twap_test["completion_score_bps"]["mean"])
    lines = [
        "# AIDEN execution research: measured results", "",
        f"The PPO actor-critic was trained for **{total_steps:,} actual interactions** "
        f"across {len(manifests)} independently initialized seeds on the "
        f"{training_meta['day_count']} real training days. "
        f"Full February validation selected **{selected}**. The frozen checkpoint "
        f"was then evaluated on all **{metadata['episode_count']} March episodes**.", "",
        f"At the primary **{config['sizes'][primary]} BTC** order size and assumed "
        f"**{config['fee_bps']:g}-bp** fee, selected PPO's all-episode mean cost was "
        f"**{selected_test['completion_score_bps']['mean']:.3f} bps**, versus "
        f"**{twap_test['completion_score_bps']['mean']:.3f} bps** for TWAP. "
        f"The PPO-minus-TWAP difference was **{difference:+.3f} bps** "
        f"({'worse' if difference > 0 else 'better' if difference < 0 else 'equal'} for PPO).", "",
        "These are actual historical-replay results, not synthetic-test scores. "
        "They do not establish live profitability, endogenous market impact, or statistical significance across independent market days.", "",
        "## Data and experimental separation", "",
        "| Split | Market days | Complete episodes | Role |",
        "|---|---:|---:|---|",
    ]
    for name, role in (("training", "Fit PPO and calibrate size/thresholds"),
                       ("validation", "Select checkpoints and seed"),
                       ("testing", "Frozen final evaluation")):
        meta = summaries[name]["metadata"]
        lines.append(f"| {name} | {meta['day_count']} | {meta['episode_count']} | {role} |")
    lines.extend([
        "", "Source files are the cleaned 5-second gzip CSVs. File hashes and row counts "
        "are saved in each split's `run_metadata.json`. `frozen_selection.json` records "
        "the model/config/source hashes and the selection timestamp before test evaluation.", "",
        f"Training median displayed bid depth was "
        f"{config['training_statistics']['total_bid_depth_btc_quantiles']['p50']} BTC. "
        f"Size rule: {config['size_rule']} Every policy "
        "uses identical quantities and episodes. No future volume sets Q.", "",
        "## Optimization evidence", "",
        "The actor and critic each use two 64-unit Tanh layers through Stable-Baselines3 PPO. "
        "The implementation, loss terms, seeds, and fixed hyperparameters are recorded in "
        "the manifests and project experiment guide. This project does not claim PPO was written from scratch.", "",
        "| Seed | Actual steps | Selected checkpoint step | Completed training episodes | Distinct training episodes sampled | Parameters changed |",
        "|---:|---:|---:|---:|---:|---|",
    ])
    for manifest in manifests:
        with (run / "ppo" / f"seed_{manifest['seed']}" / "training_episodes.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        changed = manifest["initial_parameter_sha256"] != manifest["final_parameter_sha256"]
        lines.append(f"| {manifest['seed']} | {manifest['actual_timesteps']:,} "
                     f"| {manifest['best_checkpoint_timesteps']:,} | {len(rows):,} "
                     f"| {len({row['episode_index'] for row in rows}):,} | {changed} |")
    lines.extend([
        "", f"Episodes are sampled with replacement from the "
        f"{training_meta['episode_count']:,}-episode training pool; "
        "a seed need not visit every episode. Initial untrained performance is logged but "
        "is ineligible for trained checkpoint selection. All eligible checkpoints follow optimizer updates.", "",
        "![Training and validation learning curves](figures/learning_curves.png)", "",
        "## Validation selection versus held-out performance", "",
        "| PPO seed | February primary mean score (bps) | March primary mean score (bps) | Selected on February? |",
        "|---|---:|---:|---|",
    ])
    for row in frozen["candidates"]:
        score = group(test, row["policy"], primary)["completion_score_bps"]["mean"]
        lines.append(f"| {row['policy']} | {row['mean_completion_score_bps']:.3f} "
                     f"| {score:.3f} | {'Yes' if row['policy'] == selected else 'No'} |")
    lines.extend([
        "", f"March results do not change the selected seed. The {subset_size}-episode subset used "
        "for within-seed checkpoint selection is smaller than the full February ranking "
        "set, so the two reported validation means need not agree.", "",
        "| Split | Selected PPO mean score | TWAP mean score | PPO minus TWAP (bps) |",
        "|---|---:|---:|---:|",
    ])
    for split, summary in summaries.items():
        ppo_mean = group(summary, selected, primary)["completion_score_bps"]["mean"]
        twap_mean = group(summary, "twap", primary)["completion_score_bps"]["mean"]
        lines.append(f"| {split} | {ppo_mean:.3f} | {twap_mean:.3f} | {ppo_mean-twap_mean:+.3f} |")
    lines.extend([
        "", "A favorable test-day mean does not imply consistent superiority across splits. "
        "Selection chooses the best PPO seed, not the best policy including fixed controls.", "",
        "## Final test: cost, completion, and tails", "",
        "The table shows all fixed controls plus the selected PPO model. Full tables "
        "for the other seeds are in `testing/report.md`. Completed-only net IS excludes "
        "unfilled orders; the all-episode score retains their conservative residual penalty.", "",
        "| Policy | Size BTC | Complete % | Mean net IS | Median IS | P95 IS | Worst 5% IS | Mean all-episode score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in test["groups"]:
        if row["policy"].startswith("ppo_") and row["policy"] != selected:
            continue
        cost = row["net_shortfall_bps"]
        lines.append(f"| {row['policy']} | {row['parent_quantity_btc']} "
                     f"| {100*row['completion_rate']:.2f} | {number(cost['mean'])} "
                     f"| {number(cost['median'])} | {number(cost['p95'])} "
                     f"| {number(cost['worst_5pct_mean'])} "
                     f"| {row['completion_score_bps']['mean']:.3f} |")
    lines.extend([
        "", "![Cost distributions](testing/figures/cost_distribution.png)", "",
        "![Completion and cost](testing/figures/completion_vs_cost.png)", "",
        "## Paired comparison and size generalization", "",
        "| Size | PPO minus TWAP mean IS (bps), complete pairs | PPO cost win % | Complete pairs | PPO completion % | TWAP completion % |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for size in config["sizes"]:
        row, twap = group(test, selected, size), group(test, "twap", size)
        paired = row["paired_vs_twap"]
        win = paired["cost_win_rate"]
        lines.append(f"| {size} | {number(paired['net_shortfall_delta_bps']['mean'])} "
                     f"| {number(100*win if win is not None else None, 1)} "
                     f"| {paired['complete_pairs']} | {100*row['completion_rate']:.2f} "
                     f"| {100*twap['completion_rate']:.2f} |")
    lines.extend([
        "", f"The agent was trained only at the {primary} size. Other sizes "
        "measure transfer, not additional tuned agents. If completion drops, the "
        "completed-only paired cost is a selected subset and cannot represent full-order performance.", "",
        "## Behavior and attribution", "",
        "| Policy | Size | WAIT | BELOW | ON | ABOVE | Mean execution time (seconds) | Mean fraction filled on last decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in test["groups"]:
        if not row["policy"].startswith("ppo_"):
            continue
        counts = row["action_counts"]
        total = sum(counts.values())
        proportions = [100*counts[str(index)]/total if total else 0 for index in range(4)]
        lines.append(f"| {row['policy']} | {row['size']} "
                     + "".join(f"| {value:.1f}% " for value in proportions)
                     + f"| {number(row['average_execution_time_seconds']['mean'], 1)} "
                     + f"| {number(row['final_decision_quantity_fraction']['mean'])} |")
    lines.extend([
        "", "WAIT is overridden at the final decision: a wait-heavy policy is largely "
        "a terminal execution policy. Compare it with the explicit wait control before "
        "attributing results to microstructure inference. The always-above-schedule control "
        "tests whether fixed faster pacing matches or exceeds the learned policy.", "",
        "A near-constant BELOW policy remains an untested explanation of the selected "
        "agent's behavior. A constant below-schedule control was not in the frozen "
        "comparison set, so superior state adaptation over that rule is unestablished. "
        "It should be tested on additional untouched days in a future study.", "",
        "Immediate execution may request the entire parent at once, whereas normal PPO "
        "urgency is limited to twice adaptive TWAP. This makes immediate an aggressive "
        "reference with a different quantity constraint, not a claim of matched action capacity.", "",
        "![Fixed first episode inventory](testing/figures/first_episode_inventory.png)", "",
        "## Regimes, hourly blocks, and fees", "",
        "![Paired cost by size and regime](testing/figures/paired_deltas_by_regime.png)", "",
        "Regime thresholds were fit on training data and frozen. Whole-episode volatility "
        "is only an ex-post descriptor. Sparse regime groups and changing nominal BTC "
        "prices can produce very uneven held-out bucket counts; use counts in `summary.json`.", "",
        "| Primary regime | Bucket | Complete pairs | Mean PPO minus TWAP IS (bps) |",
        "|---|---|---:|---:|",
    ])
    for row in test["regime_groups"]:
        if row["policy"] == selected and row["size"] == primary:
            lines.append(f"| {row['regime_feature']} | {row['regime']} "
                         f"| {row['complete_pairs']} "
                         f"| {number(row['net_shortfall_delta_bps']['mean'])} |")
    lines.extend([
        "", "A bucket containing only one or a few episodes cannot support a stable regime conclusion.", "",
        "`robustness/` contains 0/10/20-bps repricing of the actual recorded fills "
        "and hourly paired blocks against TWAP, immediate, and wait. Repricing is valid "
        "for these fixed policies because fees do not enter observations or fill mechanics. "
        "It is not an evaluation of separately retrained fee-aware agents.", "",
        "Uniform proportional fees necessarily preserve fixed-fill score rankings: "
        "`net_score = fee_bps + (1 - fee_bps/10000) * gross_score`. "
        "The fee curves check accounting sensitivity, not independent evidence of robust alpha.", "",
        "| Fee bps | Primary PPO mean score | Primary TWAP mean score | PPO minus TWAP bps |",
        "|---:|---:|---:|---:|",
    ])
    for fee in config["stress_fee_bps"]:
        fee_groups = {row["policy"]: row for row in robustness["groups"]
                      if row["size"] == primary and row["fee_bps"] == fee}
        ppo_mean = fee_groups[selected]["completion_score_bps"]["mean"]
        twap_mean = fee_groups["twap"]["completion_score_bps"]["mean"]
        lines.append(f"| {fee:g} | {ppo_mean:.3f} | {twap_mean:.3f} | {ppo_mean-twap_mean:+.3f} |")
    lines.extend([
        "", "![Fixed-fill fee sensitivity](robustness/fee_sensitivity.png)", "",
        "| Primary hourly comparison | Hours | Mean hourly IS difference | Hours with lower PPO cost |",
        "|---|---:|---:|---:|",
    ])
    for row in robustness["hourly_comparisons"]:
        if row["size"] != primary:
            continue
        lines.append(f"| PPO minus {row['benchmark']} | {row['hours_observed']} "
                     f"| {number(row['hourly_mean_net_shortfall_delta_bps']['mean'])} "
                     f"| {100*row['fraction_hours_with_lower_mean_cost']:.1f}% |")
    largest_size = max(config["sizes"], key=config["sizes"].get)
    large = group(test, selected, largest_size)
    primary_paired = selected_test["paired_vs_twap"]
    lines.extend([
        "", "Hourly means weight hours equally; episode means weight episodes equally. "
        "They need not match. Cost comparisons with incomplete controls exclude those "
        "pairs; the hourly CSV separately retains their all-episode score differences. "
        "Twenty-four hours from one day are not twenty-four independent market days.", "",
        "## Conclusions and limits", "",
        f"- Primary PPO's mean score difference from TWAP is {difference:+.3f} bps "
        f"on this test day; it wins {100*primary_paired['cost_win_rate']:.1f}% of complete paired episodes. "
        f"Its P95 cost is {selected_test['net_shortfall_bps']['p95']:.3f} bps "
        f"versus TWAP's {twap_test['net_shortfall_bps']['p95']:.3f} bps. "
        "Mean and tail performance can disagree.",
        f"- At {large['parent_quantity_btc']} BTC, the selected policy completes "
        f"{large['completed_episodes']}/{large['episodes']} orders. Its "
        f"{large['completion_score_bps']['mean']:.3f}-bp mean conservative score "
        "includes the unsold remainder; survivor-only IS must not hide completion failures.",
        "- The experiment is implemented, trained, and evaluated on the collected data. "
        "Training losses, checkpoint hashes, per-episode results, and figure inputs are retained.",
        "- PPO should be judged against the simple controls and completion failures, not only a favorable average versus one benchmark.",
        "- Seven isolated days and one held-out test day cannot establish robust out-of-sample significance; intraday windows are correlated.",
        "- Future historical books do not respond to our orders. Displayed-depth sweep costs exclude endogenous impact, hidden liquidity, queue position, and latency.",
        "- The fee is assumed, market orders ignore lot/minimum-notional rules, and residual inventory gets zero credit in the learning score while its economic mark is reported separately.",
        "- No test-day result was used to retrain or replace the February-selected checkpoint.", "",
        "## Artifact index", "",
        "- `frozen_selection.json`: pre-test selection and integrity hashes.",
        "- `ppo/seed_*/`: actual selected checkpoints, manifests, optimization/validation logs.",
        "- `training/`, `validation/`, `testing/`: raw per-episode CSV, exact paired CSV, fixed-episode step traces, summaries, figures, and data manifests.",
        "- `robustness/`: fixed-policy fee and hourly-block sensitivity tables.",
    ])
    (run / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {run / 'RESULTS.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.run)


if __name__ == "__main__":
    main()
