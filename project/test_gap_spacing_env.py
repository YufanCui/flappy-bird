"""Tests for GapSpacingEnv. Run: python test_gap_spacing_env.py"""
import unittest
import numpy as np
from gap_spacing_env import GapSpacingEnv
from original_reward_env import OriginalRewardSpacingEnv
from test_random_spacing_env import policy


def run(env, seed, steps=3000):
    obs, _ = env.reset(seed=seed); out = [obs.copy()]
    for t in range(steps):
        o, r, te, tr, info = env.step(policy(env)); out.append((o.copy(), r, te, tr, info['score']))
        if te:
            break
    return out


class GapSpacingTests(unittest.TestCase):
    def test_gap100_reproduces_spacing_env_exactly(self):
        n = 0
        for seed in range(10):
            a = run(GapSpacingEnv((100, 100), (120, 180)), seed)
            b = run(OriginalRewardSpacingEnv((120, 180)), seed)
            self.assertEqual(len(a), len(b))
            np.testing.assert_array_equal(a[0], b[0])
            for x, y in zip(a[1:], b[1:]):
                np.testing.assert_array_equal(x[0], y[0]); self.assertEqual(x[1:], y[1:])
            n += len(a)
        print(f'\n  {n} steps identical to OriginalRewardSpacingEnv', end=' ')

    def test_gap_samples_and_geometry(self):
        e = GapSpacingEnv((80, 120), (120, 180))
        for seed in range(30):
            run(e, seed, 1500)
            for p in e.pipe_geometry():
                self.assertTrue(80 <= p['gap'] <= 120)
                self.assertIn(p['top'] + p['gap'] / 2, [c + 50 + int(e._ground['y'] * .2) for c in range(20, 100, 10)])
        e.reset(seed=0); draws = [e._get_random_pipe() for _ in range(20000)]
        self.assertEqual(set(e.gap_history), set(range(80, 121)))

    def test_centers_shared_across_treatments(self):
        a, b = GapSpacingEnv((80, 120), (120, 180)), GapSpacingEnv((100, 100), (150, 150))
        a.reset(seed=5); b.reset(seed=5)
        ca = [p['top'] + p['gap'] / 2 for p in a.pipe_geometry()]
        cb = [p['top'] + p['gap'] / 2 for p in b.pipe_geometry()]
        self.assertEqual(ca, cb)
        self.assertNotEqual([p['gap'] for p in a.pipe_geometry()], [p['gap'] for p in b.pipe_geometry()])

    def test_reproducible(self):
        a = run(GapSpacingEnv((80, 120), (120, 180)), 9); b = run(GapSpacingEnv((80, 120), (120, 180)), 9)
        for x, y in zip(a[1:], b[1:]):
            np.testing.assert_array_equal(x[0], y[0]); self.assertEqual(x[1:], y[1:])


if __name__ == '__main__':
    unittest.main(verbosity=2)
