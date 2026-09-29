"""Fixed execution schedules and timing controls using shared accounting.

Call ``env.step(baseline_request(name, env))`` until ``env.done``. A baseline
returns a BTC request; depth limits, fees, partial fills, and deadline handling
remain the responsibility of the shared environment.
"""

from env import EPISODE_STEPS, SellExecutionEnv, available_bid_depth


BASELINE_NAMES = ("immediate", "twap", "liquidity", "wait", "above_schedule")


def baseline_request(name: str, env: SellExecutionEnv) -> float:
    """Return the next requested BTC quantity without changing environment state.

    These are fixed reference policies, not tuned execution algorithms. They
    inspect remaining inventory/time and, for liquidity, current and arrival
    displayed depth only. No future book or episode-wide statistic is read.
    """
    if name not in BASELINE_NAMES:
        raise ValueError(f"baseline must be one of {BASELINE_NAMES}, got {name!r}")
    if env.done:
        raise RuntimeError("cannot request a baseline action after the episode ends")

    remaining = env.remaining_btc
    if name == "wait":
        # Submit zero until the shared environment forces the deadline request.
        # This scalar control isolates the return from terminal timing: an RL
        # policy choosing WAIT throughout must match it, including residuals.
        return 0.0
    if name == "immediate":
        # Sell as much as possible now. A scalar inventory request is sufficient;
        # env.step walks the current 25 bid levels. If depth is insufficient,
        # explicitly request the residual again at the next decision. This is
        # the aggressive completion and arrival-execution reference.
        return remaining

    # Remaining-time TWAP: q = remaining inventory / remaining opportunities.
    # Recomputing from actual fills automatically spreads any earlier shortfall
    # over the remaining decisions, rather than losing it from the schedule.
    opportunities = EPISODE_STEPS - env.step_index
    scheduled = remaining / opportunities
    if name == "twap":
        return scheduled
    if name == "above_schedule":
        # Fixed 2x remaining-time TWAP, clipped to inventory with scalar min().
        # This matches always choosing the agent's above-schedule urgency and
        # helps distinguish learned timing from a constant aggressive action.
        return min(remaining, 2.0 * scheduled)

    # Liquidity heuristic: q = TWAP * clip(current_depth / arrival_depth, .5, 2).
    # Two O(25) reductions over the ordered bid mappings estimate displayed
    # liquidity. This tests whether selling faster in relatively deeper books
    # helps; the bounds and ratio are deliberately fixed and unoptimized.
    # When arrival depth is zero, its ratio is undefined: use neutral TWAP.
    arrival_depth = available_bid_depth(env.episode[0])
    current_depth = available_bid_depth(env.current_book)
    ratio = current_depth / arrival_depth if arrival_depth else 1.0
    multiplier = max(0.5, min(2.0, ratio))
    return min(remaining, scheduled * multiplier)
