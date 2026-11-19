"""Evaluate the user's original desktop checkpoints (June 2026) with the diagnostics_v1 logger.

No learning. The checkpoints were trained by train_flappy.py (SB3 PPO defaults, gym.make('FlappyBird-v0'),
LiDAR observation, 300k requested steps except flappy_model.zip). The reward code in effect when each
was trained is not recorded in the checkpoint; the user's June 2026 emails describe the variants.
Game score / pipes cleared do not depend on the reward, so these runs compare behavior only.

Both deterministic and stochastic (SB3 default predict(), as used in the original test_flappy.py)
action selection are run and labeled separately.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from diagnose_policies_v1 import run_episode, STEP_FIELDS

ROOT = Path(__file__).resolve().parent
HIST = ROOT.parent / 'historical_reference/original_desktop'
MODELS = ['flappy_model', 'flappy_model_300k', 'flappy_model_top_penalty_300k',
          'flappy_model_obstacle_penalty_300k']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=220000)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--max-steps', type=int, default=3000)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    paths = {m: HIST / f'{m}.zip' for m in MODELS}
    meta = {}
    for m, path in paths.items():
        model = PPO.load(path, device='cpu')
        meta[m] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), num_timesteps=int(model.num_timesteps))
    (a.out / 'config.json').write_text(json.dumps(dict(purpose='historical_checkpoint_diagnostic_no_learning',
        condition='fixed100 (original game geometry via RandomGapEnv((100,100)))',
        episode_seeds=[a.seed, a.seed + a.episodes - 1], max_episode_steps=a.max_steps,
        modes=['deterministic', 'stochastic(torch.manual_seed(episode_seed))'], models=meta), indent=2) + '\n')
    eps = []
    with (a.out / 'steps.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=STEP_FIELDS); w.writeheader()
        for m, path in paths.items():
            model = PPO.load(path, device='cpu')
            for mode in ['deterministic', 'stochastic']:
                for seed in range(a.seed, a.seed + a.episodes):
                    eps.append(run_episode(model, (100, 100), seed,
                        dict(group=m, train_seed=mode, condition='fixed100'), w, deterministic=mode == 'deterministic', max_steps=a.max_steps))
                sc = [e['game_score'] for e in eps[-a.episodes:]]
                print(m, mode, 'mean', np.mean(sc), 'median', np.median(sc), 'max', max(sc),
                      'truncated', sum(e['truncated'] for e in eps[-a.episodes:]), flush=True)
    with (a.out / 'episodes.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(eps[0])); w.writeheader(); w.writerows(eps)


if __name__ == '__main__':
    main()
