"""New research extension of the user's frozen local third-party environment."""
from pathlib import Path
import sys
import numbers

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
import numpy as np
from gymnasium import spaces
from flappy_bird_gymnasium import FlappyBirdEnv
from flappy_bird_gymnasium.envs.constants import PIPE_HEIGHT, PIPE_WIDTH, PLAYER_HEIGHT


class RandomGapEnv(FlappyBirdEnv):
    """Sample integer gap size independently for each new pipe pair.

    Hold the original gap-center distribution fixed: original top edge + 50.
    Separate random streams keep pipe centers paired across gap treatments.
    Geometry/reward/LiDAR code remains inherited from the frozen local package.
    """
    def __init__(self, gap_range=(80, 120), **kwargs):
        if len(gap_range) != 2 or any(
            isinstance(v, bool) or not isinstance(v, numbers.Integral) for v in gap_range
        ):
            raise ValueError('gap_range must be two integer pixel values')
        self.gap_range = tuple(map(int, gap_range))
        if kwargs.get('pipe_gap', 100) != 100:
            raise ValueError('Use gap_range to set size; reference center uses gap=100')
        super().__init__(**kwargs)
        lo, hi = self.gap_range
        center_min = 20 + int(self._ground['y'] * .2) + 50
        center_max = 90 + int(self._ground['y'] * .2) + 50
        if not (PLAYER_HEIGHT < lo <= hi and center_min-hi/2 >= 1
                and center_max+hi/2 < self._ground['y']-1):
            raise ValueError('Gap must fit above ground, below ceiling, and exceed bird height')
        self._gap_rng = None
        # Upstream LiDAR integer endpoint rounding can yield normalized values >1.
        # Declare actual nonnegative range; do not clip values or alter rewards.
        if self._use_lidar:
            self.observation_space = spaces.Box(0., np.inf, (180,), dtype=np.float64)
        elif self._normalize_obs:
            self.observation_space = spaces.Box(-np.inf, np.inf, (12,), dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        if seed is not None or self._gap_rng is None:
            # Env's own RNG is reserved for the inherited center distribution.
            super(FlappyBirdEnv, self).reset(seed=seed)
            self._gap_rng = np.random.default_rng(
                np.random.SeedSequence([self.np_random_seed, 0x474150]))
        # Reset two inherited transient animation/physics flags for replayability.
        self._player_flapped = False
        from itertools import cycle
        self._player_idx_gen = cycle([0, 1, 2, 1])
        return super().reset(seed=seed, options=options)

    def _get_random_pipe(self):
        lo, hi = self.gap_range
        # Consume a draw even for fixed size so gap RNG progress is consistent.
        u = self._gap_rng.random()
        gap = lo + int(u * (hi-lo+1))
        center = int(self._ground['y'] * .2) + 50 + int(
            self.np_random.choice([20, 30, 40, 50, 60, 70, 80, 90]))
        top = center - gap/2
        x = self._screen_width + PIPE_WIDTH + self._screen_width*.2
        return [{'x': x, 'y': top-PIPE_HEIGHT}, {'x': x, 'y': top+gap}]

    def pipe_geometry(self):
        return [dict(x=float(up['x']), top=float(up['y']+PIPE_HEIGHT),
                     bottom=float(low['y']), gap=float(low['y']-up['y']-PIPE_HEIGHT))
                for up, low in zip(self._upper_pipes, self._lower_pipes)]
