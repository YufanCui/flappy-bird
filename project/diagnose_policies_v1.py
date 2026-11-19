"""Diagnostic replay v1: per-step reward cases, actions, geometry and crash categories.

No learning. Loads saved checkpoints, runs deterministic evaluation on a NEW, predeclared
development episode-seed panel, and logs per-step state so failures can be categorized.

Reward cases are reconstructed in the inherited execution order of the frozen vendor step():
  1. scored this step            -> +1
  2. else LiDAR private zone      -> -1   (any ray distance < PLAYER_PRIVATE_ZONE = 32 px)
  3. else                         -> +0.1
  4. player_y < 0 (above screen)  -> -0.5 (overrides 1-3)
  5. crash (ground or pipe)       -> -1   (overrides 1-4, terminates)
The reconstructed reward is asserted equal to the environment's reward at every step, so the
reconstruction is checked rather than assumed. Underlying flags are stored even when overridden.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import numpy as np
from random_gap_env import RandomGapEnv
import gymnasium as gym
import pygame
from stable_baselines3 import PPO
import stable_baselines3
import torch
from flappy_bird_gymnasium.envs.constants import (
    PIPE_HEIGHT, PIPE_WIDTH, PLAYER_HEIGHT, PLAYER_WIDTH, PLAYER_PRIVATE_ZONE, LIDAR_MAX_DISTANCE)

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'results/gap_retry_20261118'
GROUPS = ['fixed100', 'random90_110', 'random80_120']
CONDITIONS = {'fixed100': (100, 100), 'random80_120': (80, 120)}
STEP_FIELDS = ['group', 'train_seed', 'condition', 'episode_seed', 'step', 'action',
    'player_y', 'vel_y', 'rot', 'reward', 'reward_case', 'scored', 'proximity', 'proximity_source',
    'min_ray_px', 'above_screen', 'crash', 'crash_type', 'score', 'terminated', 'truncated',
    'next_pipe_x', 'next_pipe_top', 'next_pipe_bottom', 'next_pipe_gap', 'bird_center_offset']


def next_pipe(env):
    """First pipe whose right edge is not yet behind the bird's left edge (simulator geometry)."""
    pipes = sorted(env.pipe_geometry(), key=lambda p: p['x'])
    for p in pipes:
        if p['x'] + PIPE_WIDTH >= env._player_x:
            return p
    return pipes[-1]


def crash_type(env):
    """Categorize the terminal collision using the same rects as the vendor _check_crash()."""
    if env._player_y + PLAYER_HEIGHT >= env._ground['y'] - 1:
        return 'ground'
    bird = pygame.Rect(env._player_x, env._player_y, PLAYER_WIDTH, PLAYER_HEIGHT)
    for up, low in zip(env._upper_pipes, env._lower_pipes):
        if bird.colliderect(pygame.Rect(up['x'], up['y'], PIPE_WIDTH, PIPE_HEIGHT)):
            return 'upper_pipe'
        if bird.colliderect(pygame.Rect(low['x'], low['y'], PIPE_WIDTH, PIPE_HEIGHT)):
            return 'lower_pipe'
    return 'unknown'


def run_episode(model, gap, seed, tags, writer, deterministic=True, max_steps=3000):
    if not deterministic:  # stochastic diagnostic: reproducible action sampling per episode
        torch.manual_seed(seed)
    e = gym.wrappers.TimeLimit(RandomGapEnv(gap), max_episode_steps=max_steps)
    env = e.unwrapped
    obs, info = e.reset(seed=seed)
    prev_score, ret = 0, 0.
    rows = []
    for step in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)
        obs, r, t, tr, info = e.step(action)
        ret += r
        # Recompute pixel distances exactly as vendor LIDAR.scan() did (obs*max loses exactness at 32 px).
        c = env._lidar.collisions
        raw = np.sqrt((env._player_x + PLAYER_WIDTH - c[:, 0]) ** 2
                      + (env._player_y + PLAYER_HEIGHT / 2 - c[:, 1]) ** 2)
        assert np.allclose(raw / LIDAR_MAX_DISTANCE, obs)
        i = int(np.argmin(raw))
        prox = bool(raw[i] < PLAYER_PRIVATE_ZONE)
        prox_src = ''
        if prox:
            prox_src = 'ground' if env._lidar.collisions[i][1] >= env._ground['y'] else 'pipe'
        scored = info['score'] > prev_score
        above = env._player_y < 0
        crash = bool(t) and crash_type(env) != 'unknown'
        if crash: case = 'crash'
        elif above: case = 'above_screen'
        elif scored: case = 'score'
        elif prox: case = 'proximity'
        else: case = 'alive'
        expected = {'crash': -1, 'above_screen': -.5, 'score': 1, 'proximity': -1, 'alive': .1}[case]
        assert expected == r, (tags, seed, step, case, r)
        p = next_pipe(env)
        row = dict(**tags, episode_seed=seed, step=step, action=action, player_y=env._player_y,
            vel_y=env._player_vel_y, rot=env._player_rot, reward=r, reward_case=case,
            scored=scored, proximity=prox, proximity_source=prox_src, min_ray_px=round(float(raw[i]), 3),
            above_screen=above, crash=crash, crash_type=crash_type(env) if crash else '',
            score=info['score'], terminated=t, truncated=tr, next_pipe_x=p['x'],
            next_pipe_top=p['top'], next_pipe_bottom=p['bottom'], next_pipe_gap=p['gap'],
            bird_center_offset=(env._player_y + PLAYER_HEIGHT / 2) - (p['top'] + p['bottom']) / 2)
        writer.writerow(row); rows.append(row)
        prev_score = info['score']
        if t or tr:
            break
    e.close()
    cases = [x['reward_case'] for x in rows]
    last = rows[-1]
    return dict(**tags, episode_seed=seed, game_score=info['score'], episode_return=round(ret, 10),
        steps=len(rows), terminated=last['terminated'], truncated=last['truncated'],
        crash_type=last['crash_type'], death_pipe_gap=last['next_pipe_gap'] if last['crash'] else '',
        death_center_offset=round(last['bird_center_offset'], 2) if last['crash'] else '',
        flap_rate=round(np.mean([x['action'] for x in rows]), 4),
        n_alive=cases.count('alive'), n_score=cases.count('score'), n_proximity=cases.count('proximity'),
        n_above_screen=cases.count('above_screen'), n_crash=cases.count('crash'),
        n_prox_pipe=sum(x['proximity_source'] == 'pipe' for x in rows),
        n_prox_ground=sum(x['proximity_source'] == 'ground' for x in rows),
        n_prox_flag_any=sum(x['proximity'] for x in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=220000, help='first development episode seed')
    p.add_argument('--episodes', type=int, default=20)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    (a.out / 'source').mkdir()
    for f in ['diagnose_policies_v1.py', 'random_gap_env.py']:
        shutil.copy2(ROOT / f, a.out / 'source' / f)
    models = {f'{g}_seed{s}': RUN / f'{g}_seed{s}.zip' for g in GROUPS for s in (0, 1, 2)}
    config = dict(purpose='development_diagnostic_v1_no_learning', deterministic=True,
        episode_seeds=[a.seed, a.seed + a.episodes - 1], episodes_per_condition=a.episodes,
        conditions=CONDITIONS, max_episode_steps=3000, learning_updates=0,
        panel_note='New development panel; does not use proposed validation 200000-200099 or final 300000-300099.',
        models={k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in models.items()},
        source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.out / 'source').iterdir()},
        versions=dict(python=platform.python_version(), sb3=stable_baselines3.__version__,
            torch=torch.__version__, gymnasium=gym.__version__, numpy=np.__version__, pygame=pygame.version.ver),
        private_zone_px=PLAYER_PRIVATE_ZONE, lidar_max_px=LIDAR_MAX_DISTANCE)
    (a.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    episodes = []
    with (a.out / 'steps.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=STEP_FIELDS); w.writeheader()
        for name, path in models.items():
            model = PPO.load(path, device='cpu')
            g, s = name.rsplit('_seed', 1)
            for cond, gap in CONDITIONS.items():
                for seed in range(a.seed, a.seed + a.episodes):
                    episodes.append(run_episode(model, gap, seed,
                        dict(group=g, train_seed=int(s), condition=cond), w))
                print(name, cond, 'mean score', np.mean([x['game_score'] for x in episodes[-a.episodes:]]), flush=True)
    with (a.out / 'episodes.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(episodes[0])); w.writeheader(); w.writerows(episodes)
    print('Output:', a.out)


if __name__ == '__main__':
    main()
