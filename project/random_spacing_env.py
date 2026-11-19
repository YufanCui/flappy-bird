"""Random horizontal pipe spacing: extension of the user's frozen local environment.

Teacher's requirement (verbal, as relayed by the user on 2026-11-18): make the distance between
consecutive pipe pairs a random variable instead of a fixed value.

Original game (frozen vendor): three pipe pairs recycle; an exiting pair respawns at x=397.6.
With 4-px steps this yields a deterministic repeating spacing pattern 144, 144, 164 px
(mean ~150.7), i.e. no randomness in horizontal spacing.

RandomSpacingEnv: initial pairs at x0=288, x0+D1, x0+D1+D2; each respawned pair is placed at
(rightmost other pair's x) + D, with integer D drawn uniformly from spacing_range, once per new pair.
Vertical gap stays 100; gap centers keep the inherited distribution and RNG (env.np_random);
D uses a separate seeded stream, so different spacing treatments see identical center sequences.

Equivalence argument: the vendor step() respawns the pair at x=397.6, then computes LiDAR,
reward and crash. LiDAR rays start at the bird's nose (x=91) with range 196 px, so they reach at
most x=287; collisions involve the bird (x 57..91). A pair at x>=288 therefore cannot affect that
step's observation, reward or termination. We move the respawned pair after step() returns and
assert the new position is also >= 288, so the step's outputs are exactly what they would be had
the pair been spawned there directly.
"""
from itertools import cycle
from pathlib import Path
import numbers
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
import numpy as np
from gymnasium import spaces
from flappy_bird_gymnasium import FlappyBirdEnv
from flappy_bird_gymnasium.envs.constants import PIPE_HEIGHT, LIDAR_MAX_DISTANCE, PLAYER_WIDTH

ORIGINAL_MEAN_SPACING = 150.7  # measured from the frozen vendor (pattern 144,144,164)


class RandomSpacingEnv(FlappyBirdEnv):
    def __init__(self, spacing_range=(120, 180), **kwargs):
        if len(spacing_range) != 2 or any(
                isinstance(v, bool) or not isinstance(v, numbers.Integral) for v in spacing_range):
            raise ValueError('spacing_range must be two integer pixel values')
        lo, hi = map(int, spacing_range)
        if kwargs.get('pipe_gap', 100) != 100:
            raise ValueError('Vertical gap is held at the original 100 px')
        super().__init__(**kwargs)
        # Placement must stay off screen / beyond LiDAR when three pairs recycle:
        # new x = exiting(~-56) + D1 + D2 + D3 >= 288 needs 3*lo > 344.
        if not (lo <= hi and 3 * lo - 56 >= self._screen_width):
            raise ValueError('spacing_range lower bound too small (need >= 115)')
        self.spacing_range = (lo, hi)
        self._spacing_rng = None
        self.spacing_history = []
        # Same observation-space bound correction as RandomGapEnv (values are not clipped).
        if self._use_lidar:
            self.observation_space = spaces.Box(0., np.inf, (180,), dtype=np.float64)
        elif self._normalize_obs:
            self.observation_space = spaces.Box(-np.inf, np.inf, (12,), dtype=np.float64)

    def _draw_spacing(self):
        lo, hi = self.spacing_range
        d = lo + int(self._spacing_rng.random() * (hi - lo + 1))  # consume a draw even if fixed
        self.spacing_history.append(d)
        return d

    def reset(self, *, seed=None, options=None):
        if seed is not None or self._spacing_rng is None:
            super(FlappyBirdEnv, self).reset(seed=seed)
            self._spacing_rng = np.random.default_rng(
                np.random.SeedSequence([self.np_random_seed, 0x535041]))
        self._player_flapped = False
        self._player_idx_gen = cycle([0, 1, 2, 1])
        self.spacing_history = []
        obs, info = super().reset(seed=seed, options=options)
        x = float(self._upper_pipes[0]['x'])
        for up, low in zip(self._upper_pipes[1:], self._lower_pipes[1:]):
            x += self._draw_spacing()
            up['x'] = low['x'] = x
        assert self._upper_pipes[1]['x'] > self._lidar_reach()
        obs, _ = self._get_observation()  # pairs 2-3 were already out of LiDAR range; recompute anyway
        return obs, info

    def _lidar_reach(self):
        return self._player_x + PLAYER_WIDTH + LIDAR_MAX_DISTANCE

    def step(self, action):
        before = [p['x'] for p in self._upper_pipes]
        out = super().step(action)
        after = [p['x'] for p in self._upper_pipes]
        for i, (b, a) in enumerate(zip(before, after)):
            if a > b:  # pair i was respawned this step by the vendor at x=397.6
                assert a > self._lidar_reach()
                new_x = max(x for j, x in enumerate(after) if j != i) + self._draw_spacing()
                assert new_x >= self._screen_width
                self._upper_pipes[i]['x'] = self._lower_pipes[i]['x'] = new_x
                after[i] = new_x
        return out

    def pipe_geometry(self):
        return [dict(x=float(up['x']), top=float(up['y'] + PIPE_HEIGHT),
                     bottom=float(low['y']), gap=float(low['y'] - up['y'] - PIPE_HEIGHT))
                for up, low in zip(self._upper_pipes, self._lower_pipes)]
