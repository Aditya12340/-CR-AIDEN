"""Thin Gymnasium adapter; all execution and rewards live in env.py."""

from dataclasses import asdict
from numbers import Integral

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env import SellExecutionEnv
from features import FEATURE_NAMES, observation_vector


class GymSellExecutionEnv(gym.Env):
    """Discrete(4), ten float32 observations, and a finite execution horizon.

    Supply episodes from ONE chosen split. reset(seed=...) controls episode
    sampling; reset(options={"episode_index": i}) selects a paired evaluation
    episode explicitly. Parent size and fee assumptions are fixed for this env.
    """

    metadata = {"render_modes": []}

    def __init__(self, episodes, parent_quantity_btc: float, *, fee_bps: float = 0.0):
        super().__init__()
        self.episodes = tuple(episodes)
        if not self.episodes:
            raise ValueError("at least one complete episode is required")
        self.parent_quantity_btc = parent_quantity_btc
        self.fee_bps = fee_bps
        self.core = None
        self.episode_index = None
        self.action_space = spaces.Discrete(4)
        # These are natural bounds, not empirical quantiles or test-fitted scales.
        # Returns and log volume/depth have no defensible small universal bound.
        low = [0, 0, 0, -1, 0, -np.inf, -np.inf, 0, 0, -1]
        high = [1, 1, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, 1]
        self.observation_space = spaces.Box(
            np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32),
            shape=(len(FEATURE_NAMES),), dtype=np.float32,
        )

    def _observation(self):
        result = np.asarray(observation_vector(self.core.observation()), dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("observation cannot be represented as finite float32 values")
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"episode_index"}:
            raise ValueError("only episode_index is supported in reset options")
        index = options.get("episode_index")
        if index is None:
            index = int(self.np_random.integers(len(self.episodes)))
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError("episode_index must be an integer")
        if not 0 <= index < len(self.episodes):
            raise ValueError("episode_index is out of range")
        self.episode_index = int(index)
        self.core = SellExecutionEnv(
            self.episodes[index], self.parent_quantity_btc, fee_bps=self.fee_bps,
        )
        return self._observation(), {
            "episode_index": self.episode_index,
            "parent_quantity_btc": self.core.parent_quantity_btc,
            "arrival_midprice": self.core.arrival_midprice,
            "fee_bps": self.core.fee_bps,
        }

    def step(self, action):
        if self.core is None:
            raise RuntimeError("call reset before step")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError("action must be an integer ID from 0 through 3")
        result = self.core.step_action(int(action))
        info = {"episode_index": self.episode_index, "execution": asdict(result)}
        if result.done:
            info["summary"] = asdict(self.core.summary())
        # The deadline is part of the task, not an external training time limit.
        # Completion and exhausted horizon therefore both terminate the MDP.
        return self._observation(), result.reward, result.done, False, info
