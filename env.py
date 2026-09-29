"""Execution mechanics for five-minute BTC sell episodes.

The environment replays historical books without changing future liquidity. It
models marketable sells against the 25 displayed bid levels only; it does not
model queue position, passive orders, or our own market impact. A configurable
fee in basis points is charged on executed notional, with zero as the default.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum
from math import fsum, isfinite
from numbers import Integral
import re
from types import MappingProxyType
from typing import Any

from features import build_observation


BOOK_LEVELS = 25
EPISODE_STEPS = 60
STEP_SECONDS = 5


def _finite_float(value: Any, name: str) -> float:
    """Convert a scalar to float and reject booleans, NaN, and infinity."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc

    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number")

    return number


def _validated_bid_levels(
    book_row: Mapping[str, Any],
) -> list[tuple[float, float]]:
    """Return validated ``(price, amount)`` pairs from best to worst bid."""
    return _validated_levels(book_row, "bids")


def _validated_levels(
    book_row: Mapping[str, Any], side: str,
) -> list[tuple[float, float]]:
    """Validate all 25 distinct price levels, allowing zero displayed size."""
    bids: list[tuple[float, float]] = []
    previous_price: float | None = None

    for level in range(BOOK_LEVELS):
        price_key = f"{side}[{level}].price"
        amount_key = f"{side}[{level}].amount"

        try:
            level_price = _finite_float(book_row[price_key], price_key)
            available_btc = _finite_float(book_row[amount_key], amount_key)
        except KeyError as exc:
            raise ValueError(f"missing book field: {exc.args[0]}") from exc

        if level_price <= 0:
            raise ValueError(f"{price_key} must be positive")
        if available_btc < 0:
            raise ValueError(f"{amount_key} must be nonnegative")
        if previous_price is not None:
            if side == "bids" and level_price >= previous_price:
                raise ValueError("bid prices must be strictly decreasing by level")
            if side == "asks" and level_price <= previous_price:
                raise ValueError("ask prices must be strictly increasing by level")

        bids.append((level_price, available_btc))
        previous_price = level_price

    return bids


def available_bid_depth(book_row: Mapping[str, Any]) -> float:
    """Return total displayed BTC across the 25 validated bid levels."""
    return fsum(amount for _, amount in _validated_bid_levels(book_row))


def book_midprice(book_row: Mapping[str, Any]) -> float:
    """Return the top-of-book midprice in USDT per BTC."""
    bids = _validated_bid_levels(book_row)

    try:
        best_ask = _finite_float(book_row["asks[0].price"], "asks[0].price")
    except KeyError as exc:
        raise ValueError("missing book field: asks[0].price") from exc

    best_bid = bids[0][0]
    if best_ask <= best_bid:
        raise ValueError("best ask must be greater than best bid")

    return (best_bid + best_ask) / 2.0


def execute_sell(
    book_row: Mapping[str, Any],
    requested_btc: float,
) -> tuple[float, float, float]:
    """Execute a marketable BTC sell against the displayed bid depth.

    Returns ``(filled_btc, received_usdt, unfilled_btc)``. Proceeds are
    gross of fees, and liquidity beyond the 25 recorded levels is unavailable.
    """
    requested = _finite_float(requested_btc, "requested_btc")
    if requested < 0:
        raise ValueError("requested_btc must be nonnegative")

    remaining_btc = requested
    fills = []
    proceeds = []

    for level_price, available_btc in _validated_bid_levels(book_row):
        if remaining_btc <= 0:
            break

        fill_btc = min(remaining_btc, available_btc)
        fills.append(fill_btc)
        proceeds.append(fill_btc * level_price)
        # Recompute from the accurately summed fills: repeated subtraction can
        # otherwise leave a false residual when a request exactly equals depth.
        remaining_btc = max(0.0, requested - fsum(fills))

    # Never round a real residual into a fill: even a tiny order needs liquidity.
    filled_btc = min(requested, fsum(fills))
    return filled_btc, fsum(proceeds), max(0.0, requested - filled_btc)


def _parse_decision_time(value: Any) -> datetime:
    """Parse one timezone-aware decision timestamp."""
    if hasattr(value, "to_pydatetime"):
        if getattr(value, "nanosecond", 0):
            raise ValueError("timestamps must not contain sub-microsecond precision")
        value = value.to_pydatetime()
    elif isinstance(value, str):
        if re.search(r"[.,]\d{7,}", value):
            raise ValueError("timestamps must not contain sub-microsecond precision")
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid decision_time: {value}") from exc

    if not isinstance(value, datetime):
        raise ValueError("decision_time must be a datetime or ISO datetime string")
    if value.utcoffset() is None:
        raise ValueError("decision_time must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError("decision_time must be in UTC")

    return value


def _episode_records(episode: Any) -> list[Mapping[str, Any]]:
    """Convert a DataFrame or sequence of rows into immutable mappings."""
    if hasattr(episode, "columns") and hasattr(episode, "to_dict"):
        raw_rows = episode.to_dict(orient="records")
    elif isinstance(episode, Sequence) and not isinstance(episode, (str, bytes)):
        raw_rows = list(episode)
    else:
        raise TypeError("episode must be a DataFrame or a sequence of mappings")

    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"episode row {index} is not a mapping")
        rows.append(MappingProxyType(dict(row)))

    return rows


def validate_episode(episode: Any) -> tuple[Mapping[str, Any], ...]:
    """Validate and return one complete, UTC-aligned five-minute episode."""
    rows = _episode_records(episode)
    if len(rows) != EPISODE_STEPS:
        raise ValueError(f"episode must contain exactly {EPISODE_STEPS} rows")

    decision_times: list[datetime] = []
    validated_rows = []
    previous_book_time = None
    for index, row in enumerate(rows):
        try:
            decision_time = _parse_decision_time(row["decision_time"])
        except KeyError as exc:
            raise ValueError(f"episode row {index} is missing decision_time") from exc

        normalized = dict(row)
        for side in ("bids", "asks"):
            for level, (price, amount) in enumerate(_validated_levels(row, side)):
                normalized[f"{side}[{level}].price"] = price
                normalized[f"{side}[{level}].amount"] = amount
        book_midprice(normalized)
        try:
            book_time = _parse_decision_time(row["book_local_timestamp"])
            volume = _finite_float(row["trade_volume_btc_5s"], "trade_volume_btc_5s")
            signed = _finite_float(
                row["signed_trade_volume_btc_5s"], "signed_trade_volume_btc_5s"
            )
        except KeyError as exc:
            raise ValueError(f"episode row {index} missing {exc.args[0]}") from exc
        if not timedelta(0) <= decision_time - book_time <= timedelta(seconds=5):
            raise ValueError(f"episode row {index}: book must be known and at most 5s old")
        if previous_book_time is not None and book_time < previous_book_time:
            raise ValueError("book receipt timestamps must not decrease")
        if volume < 0 or abs(signed) > volume + 1e-12 * volume:
            raise ValueError("trade volume must be nonnegative and bound signed volume")
        normalized.update(
            decision_time=decision_time, book_local_timestamp=book_time,
            trade_volume_btc_5s=volume, signed_trade_volume_btc_5s=signed,
        )
        validated_rows.append(MappingProxyType(normalized))
        previous_book_time = book_time
        decision_times.append(decision_time)

    first_time = decision_times[0]
    if first_time.minute % 5 != 0 or first_time.second != 0 or first_time.microsecond != 0:
        raise ValueError("episode must start on a UTC-aligned five-minute boundary")

    expected_step = timedelta(seconds=STEP_SECONDS)
    for previous, current in zip(decision_times, decision_times[1:]):
        if current - previous != expected_step:
            raise ValueError("episode decision times must be consecutive five-second steps")

    return tuple(validated_rows)


class Action(IntEnum):
    """The four urgency choices from the design brief; IDs are checkpoint-stable."""

    WAIT = 0
    BELOW_SCHEDULE = 1
    ON_SCHEDULE = 2
    ABOVE_SCHEDULE = 3


@dataclass(frozen=True)
class ActionRequest:
    """Auditable conversion from an integer action to a marketable BTC request."""

    action: Action
    schedule_btc: float
    strategy_requested_btc: float
    submitted_btc: float
    available_bid_depth_btc: float
    executable_btc: float
    forced_terminal_liquidation: bool


def compile_action(
    action: Action | int, remaining_btc: float, step_index: int,
    book_row: Mapping[str, Any], *, force_terminal_liquidation: bool = True,
) -> ActionRequest:
    """Convert urgency to BTC using remaining inventory / remaining decisions.

    The four branches are O(1); the depth scan is O(25). Request and executable
    size are separate so insufficient depth remains visible as an unfilled order.
    No future volume, prices, or book sizes enter this calculation.
    """
    if isinstance(action, bool) or not isinstance(action, Integral):
        raise ValueError("action must be an integer ID from 0 through 3")
    try:
        action = Action(action)
    except ValueError as exc:
        raise ValueError("action must be an integer ID from 0 through 3") from exc
    if isinstance(step_index, bool) or not isinstance(step_index, Integral):
        raise ValueError("step_index must be an integer from 0 through 59")
    if not 0 <= step_index < EPISODE_STEPS:
        raise ValueError("step_index must be an integer from 0 through 59")
    remaining = _finite_float(remaining_btc, "remaining_btc")
    if remaining < 0:
        raise ValueError("remaining_btc must be nonnegative")
    if not isinstance(force_terminal_liquidation, bool):
        raise ValueError("force_terminal_liquidation must be a boolean")
    schedule = remaining / (EPISODE_STEPS - step_index)

    if action == Action.WAIT:
        # Submit nothing this tick; keep BTC inventory exposed to price moves.
        # Uses only scalar state, with no resting-order queue to manage. Waiting
        # lets the policy defer execution when the current book is unattractive.
        requested = 0.0
    elif action == Action.BELOW_SCHEDULE:
        # Submit half the adaptive TWAP slice. Scalar multiplication implements
        # a slower urgency choice; next tick's remaining/n recalculates catch-up
        # after this smaller fill or an earlier depth-limited partial fill.
        requested = 0.5 * schedule
    elif action == Action.ON_SCHEDULE:
        # Submit remaining/n BTC, including the current decision in n. This is
        # an adaptive equal-slice schedule: a readable TWAP reference that
        # redistributes any missed quantity across the remaining decisions.
        requested = schedule
    else:
        # Submit twice remaining/n to accelerate when liquidity or price looks
        # favorable. Scalar clipping prevents overselling; the ordered L2 bid
        # walk in execute_sell prices the larger request through up to 25 levels.
        requested = 2.0 * schedule

    requested = min(remaining, requested)
    forced = bool(
        step_index == EPISODE_STEPS - 1 and force_terminal_liquidation
        and requested < remaining
    )
    submitted = remaining if forced else requested
    depth = available_bid_depth(book_row)
    return ActionRequest(action, schedule, requested, submitted, depth,
                         min(submitted, depth), forced)


@dataclass(frozen=True)
class StepResult:
    """Accounting produced by one environment decision."""

    step_index: int
    decision_time: datetime
    strategy_requested_btc: float
    submitted_btc: float
    filled_btc: float
    received_usdt: float
    unfilled_btc: float
    remaining_btc: float
    cumulative_filled_btc: float
    cumulative_received_usdt: float
    forced_terminal_liquidation: bool
    done: bool
    action: Action | None
    fee_usdt: float
    net_received_usdt: float
    execution_reward_bps: float
    residual_penalty_bps: float
    reward: float


@dataclass(frozen=True)
class ExecutionSummary:
    """Terminal accounting for one parent sell order."""

    parent_quantity_btc: float
    arrival_midprice: float
    filled_btc: float
    remaining_btc: float
    gross_received_usdt: float
    completion_fraction: float
    average_fill_price: float | None
    implementation_shortfall_bps: float | None
    completed: bool
    fees_usdt: float
    net_received_usdt: float
    net_average_fill_price: float | None
    net_implementation_shortfall_bps: float | None
    residual_mark_usdt: float
    marked_shortfall_bps: float
    completion_score_bps: float
    total_reward: float


class SellExecutionEnv:
    """Replay one 60-step historical episode for a BTC parent sell order.

    A strategy supplies a requested BTC quantity at each step. Requests are
    clipped to remaining parent inventory. On the final step, the default
    policy submits all remaining inventory, but execution is still limited to
    the 25 displayed bid levels. Any residual inventory remains unsold.
    """

    def __init__(
        self,
        episode: Any,
        parent_quantity_btc: float,
        *,
        force_terminal_liquidation: bool = True,
        fee_bps: float = 0.0,
    ) -> None:
        parent_quantity = _finite_float(
            parent_quantity_btc, "parent_quantity_btc"
        )
        if parent_quantity <= 0:
            raise ValueError("parent_quantity_btc must be positive")
        if not isinstance(force_terminal_liquidation, bool):
            raise ValueError("force_terminal_liquidation must be a boolean")
        fee = _finite_float(fee_bps, "fee_bps")
        if not 0 <= fee < 10_000:
            raise ValueError("fee_bps must be between 0 inclusive and 10,000 exclusive")

        self.episode = validate_episode(episode)
        self.parent_quantity_btc = parent_quantity
        self.arrival_midprice = book_midprice(self.episode[0])
        self.force_terminal_liquidation = force_terminal_liquidation
        self.fee_bps = fee
        self.initial_notional_usdt = parent_quantity * self.arrival_midprice
        if not isfinite(self.initial_notional_usdt) or self.initial_notional_usdt <= 0:
            raise ValueError("parent quantity times arrival price must be finite and positive")
        self.reset()

    def reset(self) -> dict[str, float]:
        """Reset episode accounting and return the initial observation."""
        self.step_index = 0
        self.remaining_btc = self.parent_quantity_btc
        self.cumulative_filled_btc = 0.0
        self.cumulative_received_usdt = 0.0
        self.cumulative_fees_usdt = 0.0
        self.cumulative_net_received_usdt = 0.0
        self.done = False
        self.history: list[StepResult] = []
        return self.observation()

    @property
    def current_book(self) -> Mapping[str, Any]:
        """Return the current immutable historical row."""
        if self.done:
            raise RuntimeError("episode is finished; call reset before reading a book")
        return self.episode[self.step_index]

    def observation(self) -> dict[str, float]:
        """Ten causal features plus three human-readable accounting diagnostics.

        Terminal observations use the last executed row, even on early finish;
        they never reveal the next book after liquidation. The Gym adapter
        selects only FEATURE_NAMES and applies fixed, documented unit scales.
        """
        index = self.step_index - 1 if self.done else self.step_index
        observation = build_observation(
            self.episode, index, self.remaining_btc, self.parent_quantity_btc
        )
        if self.done:
            observation["time_remaining_fraction"] = 0.0
        observation["inventory_remaining_btc"] = self.remaining_btc
        observation["time_remaining_seconds"] = (
            observation["time_remaining_fraction"] * EPISODE_STEPS * STEP_SECONDS
        )
        observation["midprice_return_from_arrival_bps"] = (
            10_000.0 * (book_midprice(self.episode[index]) / self.arrival_midprice - 1)
        )
        return observation

    def action_request(self, action: Action | int) -> ActionRequest:
        """Inspect an action's sizing without executing or changing state."""
        return compile_action(
            action, self.remaining_btc, self.step_index, self.current_book,
            force_terminal_liquidation=self.force_terminal_liquidation,
        )

    def step_action(self, action: Action | int) -> StepResult:
        """Execute one of the four urgency actions; IDs are never BTC quantities."""
        request = self.action_request(action)
        return self._step(request.strategy_requested_btc, action=request.action)

    def step(self, requested_btc: float) -> StepResult:
        """Direct BTC interface for baselines; use step_action for discrete IDs."""
        return self._step(requested_btc, action=None)

    def _step(self, requested_btc: float, *, action: Action | None) -> StepResult:
        """One execution path shared by baseline quantities and discrete actions."""
        if self.done:
            raise RuntimeError("cannot step a finished episode; call reset first")

        strategy_request = _finite_float(requested_btc, "requested_btc")
        if strategy_request < 0:
            raise ValueError("requested_btc must be nonnegative")

        current_index = self.step_index
        final_step = current_index == EPISODE_STEPS - 1
        clipped_request = min(strategy_request, self.remaining_btc)
        force_remainder = (
            final_step
            and self.force_terminal_liquidation
            and clipped_request < self.remaining_btc
        )
        submitted_btc = self.remaining_btc if force_remainder else clipped_request

        filled_btc, received_usdt, unfilled_btc = execute_sell(
            self.current_book, submitted_btc
        )

        # Unsold BTC comprises inventory not requested plus the unfilled request.
        # No tolerance can turn unsold dust into a fictional execution.
        self.remaining_btc = (self.remaining_btc - submitted_btc) + unfilled_btc
        self.cumulative_filled_btc = fsum(
            [item.filled_btc for item in self.history] + [filled_btc]
        )
        self.cumulative_received_usdt = fsum(
            [item.received_usdt for item in self.history] + [received_usdt]
        )
        fee_usdt = received_usdt * self.fee_bps / 10_000.0
        net_usdt = received_usdt - fee_usdt
        self.cumulative_fees_usdt = fsum(
            [item.fee_usdt for item in self.history] + [fee_usdt]
        )
        self.cumulative_net_received_usdt = (
            self.cumulative_received_usdt - self.cumulative_fees_usdt
        )

        self.step_index += 1
        self.done = self.remaining_btc == 0.0 or self.step_index == EPISODE_STEPS

        # Reward is in basis points of the ORIGINAL parent arrival notional.
        # At completion its sum is negative net implementation shortfall.
        execution_reward = (
            10_000.0 * (net_usdt - filled_btc * self.arrival_midprice)
            / self.initial_notional_usdt
        )
        # Conservative completion objective: unsold BTC has zero credited value
        # in the learning score, so skipping a costly fill cannot erase its cost.
        # This is NOT a claim that retained BTC is worthless. summary() reports
        # its actual marked value separately, never as received cash or a fill.
        residual_penalty = (
            10_000.0 * self.remaining_btc / self.parent_quantity_btc
            if self.done else 0.0
        )

        result = StepResult(
            step_index=current_index,
            decision_time=_parse_decision_time(
                self.episode[current_index]["decision_time"]
            ),
            strategy_requested_btc=strategy_request,
            submitted_btc=submitted_btc,
            filled_btc=filled_btc,
            received_usdt=received_usdt,
            unfilled_btc=unfilled_btc,
            remaining_btc=self.remaining_btc,
            cumulative_filled_btc=self.cumulative_filled_btc,
            cumulative_received_usdt=self.cumulative_received_usdt,
            forced_terminal_liquidation=force_remainder,
            done=self.done,
            action=action,
            fee_usdt=fee_usdt,
            net_received_usdt=net_usdt,
            execution_reward_bps=execution_reward,
            residual_penalty_bps=residual_penalty,
            reward=execution_reward - residual_penalty,
        )
        self.history.append(result)
        return result

    def summary(self) -> ExecutionSummary:
        """Separate executed cash, economic marks, and the completion objective."""
        if not self.done:
            raise RuntimeError("summary is available only after the episode finishes")

        completed = self.remaining_btc == 0.0
        average_fill_price = (
            self.cumulative_received_usdt / self.cumulative_filled_btc
            if self.cumulative_filled_btc > 0
            else None
        )
        implementation_shortfall_bps = None
        if completed and average_fill_price is not None:
            implementation_shortfall_bps = (
                10_000.0
                * (self.arrival_midprice - average_fill_price)
                / self.arrival_midprice
            )

        net_average = (
            self.cumulative_net_received_usdt / self.cumulative_filled_btc
            if self.cumulative_filled_btc > 0 else None
        )
        net_shortfall = (
            10_000.0 * (1 - net_average / self.arrival_midprice)
            if completed and net_average is not None else None
        )
        residual_mark = self.remaining_btc * book_midprice(
            self.episode[self.step_index - 1]
        )
        marked_shortfall = 10_000.0 * (
            1 - (self.cumulative_net_received_usdt + residual_mark)
            / self.initial_notional_usdt
        )
        completion_score = 10_000.0 * (
            1 - self.cumulative_net_received_usdt / self.initial_notional_usdt
        )

        return ExecutionSummary(
            parent_quantity_btc=self.parent_quantity_btc,
            arrival_midprice=self.arrival_midprice,
            filled_btc=self.cumulative_filled_btc,
            remaining_btc=self.remaining_btc,
            gross_received_usdt=self.cumulative_received_usdt,
            completion_fraction=(
                self.cumulative_filled_btc / self.parent_quantity_btc
            ),
            average_fill_price=average_fill_price,
            implementation_shortfall_bps=implementation_shortfall_bps,
            completed=completed,
            fees_usdt=self.cumulative_fees_usdt,
            net_received_usdt=self.cumulative_net_received_usdt,
            net_average_fill_price=net_average,
            net_implementation_shortfall_bps=net_shortfall,
            residual_mark_usdt=residual_mark,
            marked_shortfall_bps=marked_shortfall,
            completion_score_bps=completion_score,
            total_reward=fsum(item.reward for item in self.history),
        )
