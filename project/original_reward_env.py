"""Original-reward environment versions (v2, 2026-11-18).

The frozen vendor differs from the original flappy-bird-gymnasium 0.4.0 only in the LiDAR
proximity ("private zone") penalty: vendor -1 (user's June Experiment C), original -0.5.
These classes restore the original -0.5 without touching the vendor:

- OriginalGameEnv: the original game (spacing pattern 144/144/164, gap 100), original reward.
- OriginalRewardSpacingEnv: random horizontal spacing (RandomSpacingEnv) with original reward.

Equivalence to the recovered original file is tested in test_original_reward_env.py.
"""
from itertools import cycle
import numpy as np
from gymnasium import spaces
from random_spacing_env import RandomSpacingEnv, FlappyBirdEnv

PROXIMITY_PENALTY_ORIGINAL = -0.5


class _OriginalProximity:
    def _get_observation_lidar(self):
        distances, reward = super()._get_observation_lidar()
        if reward is not None:  # vendor returns -1 when any ray < PLAYER_PRIVATE_ZONE
            reward = PROXIMITY_PENALTY_ORIGINAL
        return distances, reward


class OriginalGameEnv(_OriginalProximity, FlappyBirdEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._use_lidar:  # same bound correction as other project envs; values not clipped
            self.observation_space = spaces.Box(0., np.inf, (180,), dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        self._player_flapped = False  # transient-state cleanup for replayability (as RandomGapEnv)
        self._player_idx_gen = cycle([0, 1, 2, 1])
        return super().reset(seed=seed, options=options)

    def pipe_geometry(self):
        from flappy_bird_gymnasium.envs.constants import PIPE_HEIGHT
        return [dict(x=float(u['x']), top=float(u['y'] + PIPE_HEIGHT), bottom=float(l['y']),
                     gap=float(l['y'] - u['y'] - PIPE_HEIGHT)) for u, l in zip(self._upper_pipes, self._lower_pipes)]


class OriginalRewardSpacingEnv(_OriginalProximity, RandomSpacingEnv):
    pass


# --- v3 main environment (2026-11-18): random vertical gap size, original spacing pattern, original reward.
from random_gap_env import RandomGapEnv  # noqa: E402


class OriginalRewardGapEnv(_OriginalProximity, RandomGapEnv):
    """Teacher's requirement: gap between upper and lower pipe is random (per new pair).

    Horizontal spacing keeps the original 144/144/164 pattern; centers keep the original distribution;
    reward is the original 0.4.0 reward (proximity -0.5). gap_range=(100,100) reproduces the original game.
    """
    pass
