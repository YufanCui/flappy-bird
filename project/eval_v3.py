"""Evaluate ONE train_v3 checkpoint. The agent parts of the spec (observation/reward) are kept; only the
environment randomness is replaced by each evaluation condition. No learning.

Usage: python eval_v3.py --run results/<run> --label gap80_120_seed0 --conditions gap100_100 gap80_120 ...
"""
import argparse
import csv
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
import env_v3
from eval_spacing_pilot import episode

FIELDS = ['label', 'spec', 'train_seed', 'checkpoint', 'condition', 'mode', 'cap', 'episode_seed', 'game_score',
          'pipes_cleared', 'steps', 'episode_return', 'terminated', 'truncated']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--checkpoint', default='final')
    p.add_argument('--conditions', nargs='+', required=True, help='randomness specs, e.g. gap80_120')
    p.add_argument('--modes', nargs='+', default=['deterministic'])
    p.add_argument('--seed0', type=int, default=230000)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--cap', type=int, default=10000, help='max steps per evaluation episode')
    p.add_argument('--tag', default='', help='suffix for the output CSV name')
    a = p.parse_args()
    torch.set_num_threads(1)
    meta = json.loads((a.run / f'{a.label}.json').read_text())
    path = a.run / f'{a.label}.zip' if a.checkpoint == 'final' else a.run / 'checkpoints' / f'{a.label}_{a.checkpoint}.zip'
    model = PPO.load(path, device='cpu')
    out = a.run / 'eval' / f'{a.label}_{a.checkpoint}{a.tag}.csv'
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise SystemExit(f'{out} exists')
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for cond in a.conditions:
            spec = env_v3.with_randomness(meta['spec'], cond)
            for mode in a.modes:
                for s in range(a.seed0, a.seed0 + a.episodes):
                    row = episode(model, lambda: env_v3.make(spec), s, mode == 'deterministic', a.cap)
                    w.writerow(dict(label=a.label, spec=meta['spec'], train_seed=meta['seed'], checkpoint=a.checkpoint,
                                    condition=cond, mode=mode, cap=a.cap, episode_seed=s, **row)); f.flush()
    print('done', out, flush=True)


if __name__ == '__main__':
    main()
