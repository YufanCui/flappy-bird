"""Check v2 envs against the recovered ORIGINAL flappy-bird-gymnasium 0.4.0 env file.

Run: python test_original_reward_env.py
"""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import random_spacing_env  # noqa: F401  (puts vendor on sys.path)
from original_reward_env import OriginalGameEnv, OriginalRewardSpacingEnv
from test_random_spacing_env import policy

ORIG = Path(__file__).resolve().parent.parent / 'historical_reference/original_package/flappy_bird_env_0.4.0_original.py'
spec = importlib.util.spec_from_file_location('orig_env_040', ORIG)
orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig)
OriginalFile = orig.FlappyBirdEnv


class OriginalRewardTests(unittest.TestCase):
    def test_original_game_trajectories_identical(self):
        steps = penalties = 0
        for seed in range(20):
            a, b = OriginalGameEnv(), OriginalFile()
            oa, _ = a.reset(seed=seed); ob, _ = b.reset(seed=seed)
            np.testing.assert_array_equal(oa, ob)
            for t in range(3000):
                act = policy(a)
                xa, xb = a.step(act), b.step(act)
                np.testing.assert_array_equal(xa[0], xb[0]); self.assertEqual(xa[1:], xb[1:])
                steps += 1; penalties += xa[1] == -0.5
                if xa[2]:
                    break
        print(f'\n  {steps} steps identical; {penalties} steps with reward -0.5', end=' ')
        self.assertGreater(penalties, 50)

    def test_spacing_env_matches_original_file_with_same_layout(self):
        steps = respawns = prox = 0
        for seed in range(10):
            e = OriginalRewardSpacingEnv((120, 180)); e.reset(seed=seed)
            shadow = OriginalFile(); shadow.reset(seed=seed)
            for t in range(3000):
                for k in ['_player_y', '_player_vel_y', '_player_rot', '_player_flapped', '_loop_iter',
                          '_score', '_player_idx']:
                    setattr(shadow, k, copy.deepcopy(getattr(e, k)))
                shadow._upper_pipes = copy.deepcopy(e._upper_pipes)
                shadow._lower_pipes = copy.deepcopy(e._lower_pipes)
                shadow._ground = copy.deepcopy(e._ground)
                act = policy(e); n = len(e.spacing_history)
                x1, x2 = e.step(act), shadow.step(act)
                np.testing.assert_array_equal(x1[0], x2[0]); self.assertEqual(x1[1:], x2[1:])
                steps += 1; respawns += len(e.spacing_history) > n; prox += x1[1] == -0.5
                if x1[2]:
                    break
        print(f'\n  {steps} steps, {respawns} respawns, {prox} proximity steps identical', end=' ')
        self.assertGreater(respawns, 20)


if __name__ == '__main__':
    unittest.main(verbosity=2)
