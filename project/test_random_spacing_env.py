"""Sanity tests for RandomSpacingEnv. Run: python test_random_spacing_env.py"""
import copy
import unittest
import numpy as np
from random_spacing_env import RandomSpacingEnv
from flappy_bird_gymnasium import FlappyBirdEnv


def policy(env):
    """Scripted controller (simulator state): flap when below the next gap's center and falling."""
    nxt = min((p for p in env.pipe_geometry() if p['x'] + 52 >= env._player_x), key=lambda p: p['x'])
    target = (nxt['top'] + nxt['bottom']) / 2 + 8
    return int(env._player_y + 12 > target and env._player_vel_y >= 0)


def spawn_centers(env):
    return [p['top'] + 50 for p in env.pipe_geometry()]


class SpacingTests(unittest.TestCase):
    def roll(self, env, steps=2000, seed=0, keep_alive=True):
        """Advance pipes; teleport the bird to a safe spot so pipes keep cycling."""
        env.reset(seed=seed)
        log = []
        for t in range(steps):
            if keep_alive:
                env._player_y, env._player_vel_y = 230, 0
                env._upper_pipes_backup = None
            before = [p['x'] for p in env._upper_pipes]
            env.step(0) if not keep_alive else self._safe_step(env)
            log.append((before, [p['x'] for p in env._upper_pipes]))
        return log

    @staticmethod
    def _safe_step(env):
        # Move pipes far below/above so the bird cannot collide: only horizontal motion matters here.
        for up, low in zip(env._upper_pipes, env._lower_pipes):
            up['y'], low['y'] = -1000.0, 1000.0
        env.step(0)

    def test_spacing_range_and_placement(self):
        for lo, hi in [(120, 180), (150, 150), (115, 300)]:
            e = RandomSpacingEnv((lo, hi))
            log = self.roll(e)
            self.assertTrue(all(lo <= d <= hi for d in e.spacing_history))
            if lo < hi:
                self.assertEqual(set(e.spacing_history), set(range(lo, hi + 1)) & set(e.spacing_history))
            for before, after in log:
                for b, a in zip(before, after):
                    if a > b:
                        self.assertGreaterEqual(a, 288)      # respawned off screen
                    else:
                        self.assertAlmostEqual(a, b - 4)     # otherwise moves exactly 4 px
            xs = sorted(log[-1][1])
            self.assertTrue(all(lo <= y - x <= hi for x, y in zip(xs, xs[1:])))

    def test_consecutive_spacing_equals_draws(self):
        e = RandomSpacingEnv((120, 180))
        e.reset(seed=3)
        xs = sorted(p['x'] for p in e._upper_pipes)
        self.assertEqual([xs[1] - xs[0], xs[2] - xs[1]], e.spacing_history[:2])
        log = self.roll(e, steps=600, seed=3)
        order, gaps = [], []
        for before, after in log:
            for b, a in zip(before, after):
                if a > b:
                    others = [x for x in after if x != a]
                    gaps.append(a - max(others))
        self.assertEqual(gaps, e.spacing_history[2:2 + len(gaps)])

    def test_full_range_coverage(self):
        e = RandomSpacingEnv((120, 180)); e.reset(seed=0)
        draws = [e._draw_spacing() for _ in range(20000)]
        self.assertEqual(set(draws), set(range(120, 181)))
        self.assertAlmostEqual(np.mean(draws), 150, delta=0.5)

    def test_reproducible_and_centers_shared(self):
        a, b, c = RandomSpacingEnv((120, 180)), RandomSpacingEnv((120, 180)), RandomSpacingEnv((150, 150))
        la, lb, lc = self.roll(a, 800, seed=7), self.roll(b, 800, seed=7), self.roll(c, 800, seed=7)
        self.assertEqual(a.spacing_history, b.spacing_history)
        self.assertEqual(la, lb)
        # Center sequence (env.np_random) is identical across spacing treatments and to the original env.
        for env in (a, c):
            env.reset(seed=11)
        ref = FlappyBirdEnv(); ref.reset(seed=11)
        self.assertEqual([p['y'] for p in a._upper_pipes], [p['y'] for p in ref._upper_pipes])
        self.assertEqual([p['y'] for p in c._upper_pipes], [p['y'] for p in ref._upper_pipes])

    def test_step_outputs_equal_vendor_with_same_layout(self):
        """Post-hoc repositioning must not change any step output.

        A shadow vendor env is forced to the exact same bird+pipe state before every step; its
        obs/reward/terminated/score must equal RandomSpacingEnv's outputs at every step.
        """
        compared = respawns = 0
        for seed in range(10):
            e = RandomSpacingEnv((120, 180)); obs, _ = e.reset(seed=seed)
            shadow = FlappyBirdEnv(); shadow.reset(seed=seed)
            for t in range(3000):
                for k in ['_player_y', '_player_vel_y', '_player_rot', '_player_flapped', '_loop_iter',
                          '_score', '_player_idx']:
                    setattr(shadow, k, copy.deepcopy(getattr(e, k)))
                shadow._upper_pipes = copy.deepcopy(e._upper_pipes)
                shadow._lower_pipes = copy.deepcopy(e._lower_pipes)
                shadow._ground = copy.deepcopy(e._ground)
                action = policy(e)
                n_before = len(e.spacing_history)
                o1, r1, t1, tr1, i1 = e.step(action)
                respawns += len(e.spacing_history) > n_before
                o2, r2, t2, tr2, i2 = shadow.step(action)
                np.testing.assert_array_equal(o1, o2)
                self.assertEqual((r1, t1, tr1, i1), (r2, t2, tr2, i2))
                compared += 1
                if t1:
                    break
        print(f'\n  compared {compared} steps incl. {respawns} respawn steps', end=' ')
        self.assertGreater(respawns, 20)

    def test_fixed_spacing_is_constant(self):
        e = RandomSpacingEnv((150, 150))
        self.roll(e, 1500)
        self.assertEqual(set(e.spacing_history), {150})

    def test_rejects_bad_ranges(self):
        for bad in [(100, 150), (150, 140), (120.0, 180)]:
            with self.assertRaises(ValueError):
                RandomSpacingEnv(bad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
