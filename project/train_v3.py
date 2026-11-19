"""Train ONE PPO model on an env_v3 spec. Usage:
python train_v3.py --out results/<run> --spec gap80_120 --seed 0 --steps 300000
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch
import env_v3

ROOT = Path(__file__).resolve().parent
CAP = 10000
SOURCES = ['train_v3.py', 'env_v3.py', 'original_reward_env.py', 'random_gap_env.py', 'gap_spacing_env.py',
           'random_spacing_env.py']


class GapCurriculum(BaseCallback):
    """Curriculum on the training gap range only: starts at (hi,hi) (widest fixed gap, easiest) and lowers
    the lower bound linearly to the target lo over `span` steps; then stays at the target (lo,hi).
    The new range applies to pipes generated after the change. Evaluation is always on the target task."""
    def __init__(self, env, span, out, label):
        super().__init__(); self.env, self.span, self.out, self.label = env, span, out, label
        self.lo, self.hi = env.gap_range; self.last = None

    def _on_step(self):
        frac = min(1.0, self.num_timesteps / self.span)
        lo = int(round(self.hi - (self.hi - self.lo) * frac))
        if lo != self.last:
            self.env.gap_range = (lo, self.hi); self.last = lo
            if lo % 10 == 0:
                with (self.out / 'curriculum.jsonl').open('a') as f:
                    f.write(json.dumps(dict(model=self.label, timesteps=self.num_timesteps, gap_range=[lo, self.hi])) + '\n')
        return True


class Checkpoints(BaseCallback):
    def __init__(self, out, label, every):
        super().__init__(); self.out, self.label, self.every, self.next = out, label, every, every
        self.t0 = time.monotonic(); self.scores = []

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                self.scores.append(float(info.get('score', 0)))
        if self.num_timesteps >= self.next:
            self.model.save(self.out / 'checkpoints' / f'{self.label}_{self.next // 1000}k')
            rec = dict(model=self.label, timesteps=self.num_timesteps, training_episodes=len(self.scores),
                       last20_training_mean_score=float(np.mean(self.scores[-20:])) if self.scores else None,
                       elapsed_seconds=round(time.monotonic() - self.t0, 1))
            with (self.out / 'progress.jsonl').open('a') as f:
                f.write(json.dumps(rec) + '\n')
            self.next += self.every
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--spec', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--steps', type=int, default=300000)
    p.add_argument('--ckpt-every', type=int, default=50000)
    p.add_argument('--ppo', default='{}', help='JSON dict of PPO kwargs overriding SB3 defaults')
    p.add_argument('--tag', default='')
    p.add_argument('--lr-linear', action='store_true', help='linearly decay learning rate from 3e-4 to 0')
    p.add_argument('--curriculum', type=float, default=0.0,
                   help='fraction of training over which the gap range widens from (hi,hi) to (lo,hi); 0 = off')
    a = p.parse_args()
    env_v3.parse(a.spec)
    label = f'{a.spec}{a.tag}_seed{a.seed}'
    final = a.out / f'{label}.zip'
    if final.exists():
        raise SystemExit(f'{final} exists')
    (a.out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    src = a.out / 'source'; src.mkdir(exist_ok=True)
    for f in SOURCES:
        if not (src / f).exists():
            shutil.copy2(ROOT / f, src / f)
    torch.set_num_threads(1)
    ppo = json.loads(a.ppo)
    if 'policy_kwargs' in ppo and 'net_arch' in ppo['policy_kwargs']:
        ppo['policy_kwargs']['net_arch'] = ppo['policy_kwargs']['net_arch']
    env = Monitor(gym.wrappers.TimeLimit(env_v3.make(a.spec), max_episode_steps=CAP),
                  filename=str(a.out / f'{label}.monitor.csv'), info_keywords=('score',))
    if a.lr_linear:
        lr0 = ppo.pop('learning_rate', 3e-4)
        ppo['learning_rate'] = lambda progress_remaining: lr0 * progress_remaining
    model = PPO('MlpPolicy', env, seed=a.seed, device='cpu', verbose=0, **ppo)
    callbacks = [Checkpoints(a.out, label, a.ckpt_every)]
    if a.curriculum > 0:
        callbacks.append(GapCurriculum(env.unwrapped, a.steps * a.curriculum, a.out, label))
    t0 = time.monotonic()
    model.learn(total_timesteps=a.steps, callback=callbacks)
    model.save(final); env.close()
    (a.out / f'{label}.json').write_text(json.dumps(dict(label=label, spec=a.spec, tag=a.tag, seed=a.seed,
        requested_steps=a.steps, actual_steps=int(model.num_timesteps), ppo_overrides=json.loads(a.ppo),
        lr_linear=a.lr_linear, curriculum_fraction=a.curriculum,
        train_episode_cap=CAP, reward='original 0.4.0 unless spec has noprox', seconds=round(time.monotonic() - t0, 1),
        sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
        source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in src.iterdir()}), indent=2) + '\n')
    print('DONE', label, flush=True)


if __name__ == '__main__':
    main()
