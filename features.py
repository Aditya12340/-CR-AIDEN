"""Ten causal state features for the five-minute sell-execution task.

``build_observation`` returns named values in interpretable units; the vector
helper applies fixed scales for a learning agent. Nothing is fitted to data.
Only rows up to the current decision are read, and rolling features start fresh
at each episode. The environment validates books and trade aggregates before
calling this module.
"""

from collections.abc import Mapping, Sequence
import math
import operator


FEATURE_NAMES = (
    "inventory_remaining_fraction",
    "time_remaining_fraction",
    "spread_bps",
    "depth_imbalance",
    "relative_bid_depth",
    "weighted_mid_deviation_bps",
    "recent_mid_return_bps",
    "trailing_volatility_bps",
    "recent_trade_volume",
    "signed_trade_flow",
)

_BPS_FEATURES = frozenset(
    (
        "spread_bps",
        "weighted_mid_deviation_bps",
        "recent_mid_return_bps",
        "trailing_volatility_bps",
    )
)
_EPISODE_STEPS = 60
_VOLATILITY_INTERVALS = 12  # At a five-second cadence, at most one minute.


def _midprice(row: Mapping[str, float]) -> float:
    return (float(row["bids[0].price"]) + float(row["asks[0].price"])) / 2.0


def build_observation(
    rows: Sequence[Mapping[str, float]],
    step_index: int,
    remaining_btc: float,
    parent_quantity_btc: float,
) -> dict[str, float]:
    """Describe the current decision using exactly ``FEATURE_NAMES``.

    Rows must contain the validated 25-level book and the backward-looking
    ``trade_volume_btc_5s`` / ``signed_trade_volume_btc_5s`` aggregates. Positive
    signed flow denotes buyer-initiated trades. No future rows or external
    normalization statistics enter this calculation.

    ``step_index`` is the current row, from zero through 59. Time remaining
    includes the current execution opportunity: it is 1.0 initially and 1/60
    before the final action. The environment sets it to zero on termination.
    """
    if isinstance(step_index, bool):
        raise ValueError("step_index must be an integer from 0 through 59")
    try:
        step_index = operator.index(step_index)
    except TypeError as exc:
        raise ValueError("step_index must be an integer from 0 through 59") from exc
    if not 0 <= step_index < min(len(rows), _EPISODE_STEPS):
        raise ValueError("step_index must identify an available row from 0 through 59")
    if not math.isfinite(parent_quantity_btc) or parent_quantity_btc <= 0.0:
        raise ValueError("parent_quantity_btc must be finite and positive")
    if not math.isfinite(remaining_btc) or not 0.0 <= remaining_btc <= parent_quantity_btc:
        raise ValueError("remaining_btc must be between zero and parent_quantity_btc")

    row = rows[step_index]
    bid = float(row["bids[0].price"])
    ask = float(row["asks[0].price"])
    mid = _midprice(row)
    bid_size = float(row["bids[0].amount"])
    ask_size = float(row["asks[0].amount"])
    bid_depth = math.fsum(float(row[f"bids[{level}].amount"]) for level in range(25))
    ask_depth = math.fsum(float(row[f"asks[{level}].amount"]) for level in range(25))
    total_depth = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / total_depth if total_depth else 0.0

    # Opposite-side weights shift the quote toward the ask when the bid queue is
    # larger. This is a weighted-mid proxy, not a fitted Stoikov microprice.
    # It adds top-of-book pressure to the full-depth imbalance above.
    top_size = bid_size + ask_size
    weighted_mid = (ask * bid_size + bid * ask_size) / top_size if top_size else mid

    # At most twelve adjacent log returns from this episode are available.
    # Realized volatility is sqrt(sum(r**2)); it is neither annualized nor a
    # de-meaned sample standard deviation. Warmup uses only observed intervals.
    first_interval = max(1, step_index - _VOLATILITY_INTERVALS + 1)
    log_returns = [
        math.log(_midprice(rows[index]) / _midprice(rows[index - 1]))
        for index in range(first_interval, step_index + 1)
    ]
    volume = float(row["trade_volume_btc_5s"])
    signed_volume = float(row["signed_trade_volume_btc_5s"])

    # Log depth/volume ratios stay useful across parent sizes and have gentler
    # tails than raw BTC quantities. Fixed transforms avoid fitting on test data.
    return {
        "inventory_remaining_fraction": remaining_btc / parent_quantity_btc,
        "time_remaining_fraction": (_EPISODE_STEPS - step_index) / _EPISODE_STEPS,
        "spread_bps": 10_000.0 * (ask - bid) / mid,
        "depth_imbalance": max(-1.0, min(1.0, imbalance)),
        "relative_bid_depth": math.log1p(bid_depth / parent_quantity_btc),
        "weighted_mid_deviation_bps": 10_000.0 * (weighted_mid - mid) / mid,
        "recent_mid_return_bps": 10_000.0 * log_returns[-1] if log_returns else 0.0,
        "trailing_volatility_bps": 10_000.0 * math.sqrt(math.fsum(r * r for r in log_returns)),
        "recent_trade_volume": math.log1p(volume / parent_quantity_btc),
        "signed_trade_flow": signed_volume / volume if volume else 0.0,
    }


def observation_vector(observation: Mapping[str, float]) -> tuple[float, ...]:
    """Return the ten features in fixed order with a fixed 100-bp unit.

    Divide only basis-point features by 100; ratios and logarithms retain their
    values. There is no clipping or data-dependent fitting. A caller can convert
    the tuple to its learning framework's array type without adding a dependency
    to this module. Named observations remain in their documented original units.
    """
    values = []
    for name in FEATURE_NAMES:
        try:
            value = float(observation[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"observation must contain a numeric {name!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"observation feature {name!r} must be finite")
        values.append(value / 100.0 if name in _BPS_FEATURES else value)
    return tuple(values)
