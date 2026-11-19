"""Two random variables: vertical gap size AND horizontal spacing (v3, 2026-11-18). Original reward.

Teacher's requirement (verbal, relayed by the user): increase environment randomness; the opening
between the upper and lower pipe (vertical gap size) should also be random.

GapSpacingEnv(gap_range, spacing_range):
- Gap size G ~ U{gap_lo..gap_hi}, drawn once per new pipe pair (own seeded stream).
- Spacing D ~ U{sp_lo..sp_hi}, drawn once per new pair (RandomSpacingEnv; own seeded stream).
- Gap center keeps the ORIGINAL distribution and RNG call (np_random.integers(0, 8), as in vendor),
  so every treatment sees the identical center sequence for a given seed.
  Boundaries: top = C - G/2, bottom = C + G/2 with C = original top + 50.
- Reward: original flappy-bird-gymnasium 0.4.0 (proximity -0.5).
With gap_range=(100,100) this env reproduces OriginalRewardSpacingEnv exactly (tested).
"""
import numbers
import gymnasium
import numpy as np
from original_reward_env import _OriginalProximity
from random_spacing_env import RandomSpacingEnv
from flappy_bird_gymnasium.envs.constants import PIPE_HEIGHT, PIPE_WIDTH, PLAYER_HEIGHT


class GapSpacingEnv(_OriginalProximity, RandomSpacingEnv):
    def __init__(self, gap_range=(80, 120), spacing_range=(120, 180), **kwargs):
        if len(gap_range) != 2 or any(isinstance(v, bool) or not isinstance(v, numbers.Integral) for v in gap_range):
            raise ValueError('gap_range must be two integer pixel values')
        super().__init__(spacing_range, **kwargs)
        lo, hi = self.gap_range = tuple(map(int, gap_range))
        ground = self._ground['y']
        cmin, cmax = 20 + int(ground * .2) + 50, 90 + int(ground * .2) + 50
        if not (PLAYER_HEIGHT < lo <= hi and cmin - hi / 2 >= 1 and cmax + hi / 2 < ground - 1):
            raise ValueError('gap must exceed bird height and fit on screen')
        self._gap_rng = None
        self.gap_history = []

    def reset(self, *, seed=None, options=None):
        if seed is not None or self._gap_rng is None:
            gymnasium.Env.reset(self, seed=seed)
            self._gap_rng = np.random.default_rng(np.random.SeedSequence([self.np_random_seed, 0x474150]))
        self.gap_history = []
        return super().reset(seed=seed, options=options)

    def _get_random_pipe(self):
        lo, hi = self.gap_range
        gap = lo + int(self._gap_rng.random() * (hi - lo + 1))  # consume a draw even if fixed
        self.gap_history.append(gap)
        index = self.np_random.integers(0, 8)  # identical call to the vendor -> identical centers
        center = [20, 30, 40, 50, 60, 70, 80, 90][index] + int(self._ground['y'] * 0.2) + 50
        top = center - gap / 2
        x = self._screen_width + PIPE_WIDTH + (self._screen_width * 0.2)
        return [{'x': x, 'y': top - PIPE_HEIGHT}, {'x': x, 'y': top + gap}]
