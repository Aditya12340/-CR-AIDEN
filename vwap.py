"""A training-only forecast-volume schedule and a separate market VWAP benchmark.

The execution policy uses a static UTC clock profile fitted on training days.
It cannot know the realized future volume of the order being executed. The
market VWAP function is strictly an after-the-fact benchmark; never feed its
output into this schedule or the agent's observations.
"""

from datetime import datetime, timedelta
from math import fsum, isfinite

from data import SPLIT_DAYS, _DAY_DATES
from env import EPISODE_STEPS, STEP_SECONDS


_PROFILE_FIELD = "volume_btc_per_5s_by_utc_minute"
_TRAINING_DATES = frozenset(_DAY_DATES[label] for label in SPLIT_DAYS["training"])
_STEP = timedelta(seconds=STEP_SECONDS)


def _nonnegative_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


def _timestamp(value):
    if not isinstance(value, datetime) or value.utcoffset() != timedelta(0):
        raise ValueError("decision times must be timezone-aware UTC datetimes")
    if value.microsecond or value.second % STEP_SECONDS:
        raise ValueError("decision times must lie on the five-second grid")
    return value


def _episode_times(episode):
    if len(episode) != EPISODE_STEPS:
        raise ValueError("a complete 60-row episode is required")
    try:
        times = [_timestamp(row["decision_time"]) for row in episode]
    except KeyError as exc:
        raise ValueError("missing decision_time") from exc
    start = times[0]
    if start.minute % 5 or start.second:
        raise ValueError("episode must begin on a UTC five-minute boundary")
    if any(stamp != start + index * _STEP for index, stamp in enumerate(times)):
        raise ValueError("episode must have 60 consecutive five-second decisions")
    return times


def fit_volume_profile(episodes) -> dict:
    """Average observed five-second BTC volume in each UTC minute, training only.

    A list of 1,440 volume buckets collects individual historical observations.
    Arithmetic bucket means estimate activity at that clock minute; no price,
    future held-out volume, or execution outcome is used. Grouping by minute
    pools the twelve five-second slots for a less sparse student-scale estimate.
    Missing minutes use the mean of all observed training rows. Counts and
    actual dates are retained so that limited historical coverage is visible.
    """
    buckets = [[] for _ in range(1440)]
    seen_starts = set()
    seen_days = set()
    episode_count = 0
    for episode in episodes:
        times = _episode_times(episode)
        if any(stamp.date() not in _TRAINING_DATES for stamp in times):
            raise ValueError("volume profile may use only the configured training days")
        if times[0] in seen_starts:
            raise ValueError("duplicate training episode in volume profile")
        seen_starts.add(times[0])
        seen_days.add(times[0].date().isoformat())
        for row, stamp in zip(episode, times):
            try:
                volume = _nonnegative_number(row["trade_volume_btc_5s"], "trade_volume_btc_5s")
            except KeyError as exc:
                raise ValueError("missing trade_volume_btc_5s") from exc
            buckets[stamp.hour * 60 + stamp.minute].append(volume)
        episode_count += 1
    counts = [len(bucket) for bucket in buckets]
    row_count = sum(counts)
    if not row_count:
        raise ValueError("at least one training episode is required")
    global_mean = fsum(fsum(bucket) for bucket in buckets) / row_count
    return {
        _PROFILE_FIELD: [fsum(bucket) / len(bucket) if bucket else global_mean for bucket in buckets],
        "observations_by_utc_minute": counts,
        "episodes": episode_count,
        "rows": row_count,
        "training_days": sorted(seen_days),
        "fallback_volume_btc_per_5s": global_mean,
    }


def schedule_weights(profile: dict, start: datetime) -> tuple[float, ...]:
    """Look up 60 static forecast weights using only the arrival clock time.

    Values are expected BTC per five-second bin, not realized future volume.
    They need not sum to one: the request function normalizes the remaining
    weights. Zero weights are valid and midnight wraps naturally in UTC.
    """
    start = _timestamp(start)
    try:
        raw = profile[_PROFILE_FIELD]
    except KeyError as exc:
        raise ValueError(f"profile must contain {_PROFILE_FIELD}") from exc
    if len(raw) != 1440:
        raise ValueError("volume profile must contain all 1440 UTC minutes")
    values = tuple(_nonnegative_number(value, "profile volume") for value in raw)
    weights = []
    for index in range(EPISODE_STEPS):
        stamp = start + index * _STEP
        weights.append(values[stamp.hour * 60 + stamp.minute])
    return tuple(weights)


def vwap_request(env, weights) -> float:
    """Request a forecast-volume share of the remaining parent, without stepping.

    For current index i, q = remaining * w[i] / sum(w[i:]). The suffix reduction
    is O(60), small enough to keep this implementation readable. Recomputing
    from actual remaining inventory redistributes partial fills over the later
    forecast weights. If all remaining weights are zero, use remaining-time
    TWAP. The shared environment enforces displayed depth, fees, and deadline
    liquidation, so the VWAP policy gains no different execution privileges.
    """
    if env.done:
        raise RuntimeError("cannot request VWAP after the episode ends")
    if len(weights) != EPISODE_STEPS:
        raise ValueError("VWAP schedule must contain 60 weights")
    values = tuple(_nonnegative_number(value, "schedule weight") for value in weights)
    index = env.step_index
    remaining = env.remaining_btc
    future_total = fsum(values[index:])
    requested = (remaining * (values[index] / future_total) if future_total > 0
                 else remaining / (EPISODE_STEPS - index))
    return min(remaining, requested)


def market_vwap(episode) -> dict:
    """Compute realized market VWAP over (arrival, final decision], 59 bins.

    Row zero summarizes trades at/before arrival and is excluded. Rows 1..59
    cover (t0, t0 + 295 seconds], the execution opportunities observable in the
    stored episode. The unrecorded interval (t295, t300] is not reconstructed.
    Sum trade notionals and BTC volumes separately, then divide; averaging bin
    VWAPs would give a quiet bin the same influence as a busy bin. A no-trade
    window has no benchmark price (None), rather than an invented quote.
    """
    _episode_times(episode)
    volumes, notionals = [], []
    for row in episode[1:]:
        try:
            volume = _nonnegative_number(row["trade_volume_btc_5s"], "trade_volume_btc_5s")
            notional = _nonnegative_number(row["trade_notional_usdt_5s"], "trade_notional_usdt_5s")
        except KeyError as exc:
            raise ValueError(f"missing market VWAP field: {exc.args[0]}") from exc
        if (volume == 0) != (notional == 0):
            raise ValueError("trade volume and notional must both be zero or both positive")
        volumes.append(volume)
        notionals.append(notional)
    volume = fsum(volumes)
    notional = fsum(notionals)
    return {
        "market_vwap_usdt_per_btc": notional / volume if volume else None,
        "market_volume_btc": volume,
        "market_notional_usdt": notional,
        "bins": EPISODE_STEPS - 1,
    }
