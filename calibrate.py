"""Freeze order sizes and descriptive regime thresholds using training days only.

Run ``python3 calibrate.py --output configs/experiment.json`` before training.
No validation/testing file is opened, and no policy is evaluated here.
"""

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from data import DATA_DIR, SPLIT_DAYS, LoadStats, load_split
from env import BOOK_LEVELS, EPISODE_STEPS, validate_episode


def quantile(values: Sequence[float], probability: float) -> float:
    """Linear interpolation between sorted samples (the usual type-7 quantile)."""
    if not values or not 0 <= probability <= 1:
        raise ValueError("quantile needs nonempty samples and probability in [0, 1]")
    ordered = sorted(values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("quantile samples must be finite")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def depth_quantiles(values: Sequence[float]) -> dict[str, float]:
    """Summarize BTC depth over all decisions in complete training episodes."""
    return {
        f"p{percent:02d}": quantile(values, percent / 100)
        for percent in (1, 10, 33, 50, 67, 90, 99)
    }


def episode_market_context(episode: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Compute arrival spread and realized episode volatility for result grouping.

    The spread is available at arrival. Volatility uses the full episode and is
    strictly a posthoc descriptor: never pass it to the policy or size an order
    from it. The 59 adjacent log returns cover the recorded 295-second span.
    Both outputs are in basis points; volatility is not annualized.
    """
    if len(episode) != EPISODE_STEPS:
        raise ValueError("market context requires a complete 60-row episode")
    mids = [
        (float(row["bids[0].price"]) + float(row["asks[0].price"])) / 2.0
        for row in episode
    ]
    if any(not math.isfinite(mid) or mid <= 0 for mid in mids):
        raise ValueError("midprices must be finite and positive")
    spread = float(episode[0]["asks[0].price"]) - float(episode[0]["bids[0].price"])
    returns = [math.log(current / previous) for previous, current in zip(mids, mids[1:])]
    return {
        "arrival_spread_bps": 10_000.0 * spread / mids[0],
        "episode_realized_volatility_bps": 10_000.0 * math.sqrt(math.fsum(r * r for r in returns)),
    }


def _file_manifest(path: Path, data_dir: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(Path("data") / path.relative_to(data_dir)),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def calibrate(data_dir: str | Path = DATA_DIR) -> dict[str, Any]:
    """Return an experiment config derived exclusively from training inputs.

    Episodes stream through validation. Only scalar depth/regime samples are
    retained for exact quantiles, keeping memory proportional to decisions and
    avoiding retention of the much larger 100-field order books.
    """
    data_dir = Path(data_dir)
    top_depths: list[float] = []
    total_depths: list[float] = []
    regimes: dict[str, list[float]] = {
        "arrival_spread_bps": [], "episode_realized_volatility_bps": [],
    }
    day_counts: Counter[str] = Counter()
    stats = LoadStats()
    for raw_episode in load_split("training", data_dir, stats=stats):
        episode = validate_episode(raw_episode)
        day_counts[episode[0]["decision_time"].date().isoformat()] += 1
        for row in episode:
            top_depths.append(row["bids[0].amount"])
            total_depths.append(math.fsum(row[f"bids[{level}].amount"] for level in range(BOOK_LEVELS)))
        for name, value in episode_market_context(episode).items():
            regimes[name].append(value)

    if len(day_counts) != len(SPLIT_DAYS["training"]):
        raise ValueError("every configured training day must have complete episodes")
    median_depth = quantile(total_depths, 0.5)
    if median_depth <= 0:
        raise ValueError("median training bid depth must be positive to calibrate order sizes")

    # Freeze one BTC quantity per regime for every policy and every split.
    # Four significant digits give readable quantities without pretending to
    # implement historical Binance lot-size/minimum-notional filters.
    sizes = {
        name: float(f"{median_depth * fraction:.4g}")
        for name, fraction in (("small", 0.01), ("medium", 0.10), ("large", 0.50))
    }
    thresholds = {
        name: {"lower": quantile(values, 1 / 3), "upper": quantile(values, 2 / 3)}
        for name, values in regimes.items()
    }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fee_bps": 10.0,
        "fee_assumption": "Illustrative constant taker fee; not verified historical Binance account pricing.",
        "sizes": sizes,
        "primary_size": "medium",
        "size_rule": "Fixed BTC quantities equal to 1%, 10%, and 50% of median 25-level training bid depth, rounded to four significant digits.",
        "seeds": [7, 19, 42],
        "total_timesteps": 200_000,
        "stress_fee_bps": [0.0, 10.0, 20.0],
        "splits": {name: list(days) for name, days in SPLIT_DAYS.items()},
        "training_statistics": {
            "episodes": stats.complete_episodes,
            "decisions": len(total_depths),
            "day_counts": dict(sorted(day_counts.items())),
            "loader_stats": asdict(stats),
            "top_bid_depth_btc_quantiles": depth_quantiles(top_depths),
            "total_bid_depth_btc_quantiles": depth_quantiles(total_depths),
        },
        "market_regimes": {
            **thresholds,
            "quantiles": [1 / 3, 2 / 3],
            "calibration_split": "training",
            "volatility_definition": "10000 * sqrt(sum(log(mid[t] / mid[t-1]) ** 2)) over 59 within-episode intervals; no annualization.",
            "usage": "Fixed descriptive grouping thresholds. Whole-episode volatility is posthoc only and never a policy input.",
        },
        "training_manifest": [
            _file_manifest(data_dir / "training" / f"{day}_5s.csv.gz", data_dir)
            for day in SPLIT_DAYS["training"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    config = calibrate(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}: {config['training_statistics']['episodes']} training episodes; BTC sizes {config['sizes']}")


if __name__ == "__main__":
    main()
