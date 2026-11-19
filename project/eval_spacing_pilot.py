"""Evaluate one spacing-pilot checkpoint on all predeclared conditions (no learning).

Usage: python eval_spacing_pilot.py --run results/spacing_pilot_v1_20261118 --label fixed150_seed0
Writes <run>/eval/<label>.csv. See PILOT_PLAN.md for the design.
"""
import argparse
import csv
from pathlib import Path
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from original_reward_env import OriginalGameEnv, OriginalRewardSpacingEnv
from flappy_bird_gymnasium.envs.constants import PIPE_WIDTH

CONDITIONS = {
    'train_fixed150': lambda: OriginalRewardSpacingEnv((150, 150)),
    'train_random120_180': lambda: OriginalRewardSpacingEnv((120, 180)),
    'ood_narrow115_125': lambda: OriginalRewardSpacingEnv((115, 125)),
    'ood_wide180_240': lambda: OriginalRewardSpacingEnv((180, 240)),
    'original_pattern': lambda: OriginalGameEnv(),
}
SEEDS = range(230000, 230020)
CAP = 10000
FIELDS = ['label', 'group', 'train_seed', 'checkpoint', 'condition', 'mode', 'episode_seed', 'game_score',
          'pipes_cleared', 'steps', 'episode_return', 'terminated', 'truncated']


def episode(model, make, seed, deterministic):
    if not deterministic:
        torch.manual_seed(seed)
    e = gym.wrappers.TimeLimit(make(), max_episode_steps=CAP)
    env = e.unwrapped
    obs, _ = e.reset(seed=seed)
    cleared, ret = 0, 0.
    for step in range(1, CAP + 1):
        before = [p['x'] for p in env._upper_pipes]
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, t, tr, info = e.step(action)
        ret += r
        for b, p in zip(before, env._upper_pipes):
            # full-body clearance: pipe's right edge moves from >= bird's left edge to < it (no respawn)
            if p['x'] < b and b + PIPE_WIDTH >= env._player_x > p['x'] + PIPE_WIDTH:
                cleared += 1
        if t or tr:
            break
    e.close()
    return dict(game_score=info['score'], pipes_cleared=cleared, steps=step, episode_return=round(ret, 6),
                terminated=t, truncated=tr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--checkpoint', default='final', help="'final' or e.g. '100k'")
    p.add_argument('--conditions', nargs='+', default=list(CONDITIONS))
    p.add_argument('--modes', nargs='+', default=['deterministic', 'stochastic'])
    p.add_argument('--episodes', type=int, default=len(SEEDS))
    a = p.parse_args()
    torch.set_num_threads(1)
    path = a.run / f'{a.label}.zip' if a.checkpoint == 'final' else a.run / 'checkpoints' / f'{a.label}_{a.checkpoint}.zip'
    model = PPO.load(path, device='cpu')
    group, seed = a.label.rsplit('_seed', 1)
    out = a.run / 'eval' / f'{a.label}_{a.checkpoint}.csv'
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise SystemExit(f'{out} exists')
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for cond in a.conditions:
            for mode in a.modes:
                for s in list(SEEDS)[:a.episodes]:
                    row = episode(model, CONDITIONS[cond], s, mode == 'deterministic')
                    w.writerow(dict(label=a.label, group=group, train_seed=seed, checkpoint=a.checkpoint,
                                    condition=cond, mode=mode, episode_seed=s, **row)); f.flush()
    print('done', out, flush=True)


if __name__ == '__main__':
    main()
