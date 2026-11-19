"""Environment factory v3 (2026-11-18). Spec string = '+'-joined parts, e.g. 'gap80_120+stack4'.

Randomness parts (what the environment samples):
  gapA_B     vertical gap size G ~ U{A..B} per new pair (main randomness; gap100_100 = original game)
  spA_B      horizontal spacing D ~ U{A..B} per new pair (extra randomness; default original 144/144/164)
Agent/training parts (what the agent sees / is rewarded for):
  stack4     stack last 4 LiDAR observations (velocity information)
  features   vendor 12-dim feature observation (privileged; has no proximity penalty by vendor design)
  noprox     remove the proximity penalty (reward change only)
Default reward: original flappy-bird-gymnasium 0.4.0 (proximity -0.5). Vendor files are never modified.
"""
import gymnasium as gym
from original_reward_env import _OriginalProximity
from random_gap_env import RandomGapEnv
from gap_spacing_env import GapSpacingEnv
from flappy_bird_gymnasium import FlappyBirdEnv

RANDOMNESS = ('gap', 'sp')
AGENT = ('stack4', 'features', 'noprox')


class _NoProximity:
    def _get_observation_lidar(self):
        distances, _ = FlappyBirdEnv._get_observation_lidar(self)
        return distances, None


def _rng(part, prefix):
    a, b = part[len(prefix):].split('_')
    return int(a), int(b)


def parse(spec):
    parts = spec.split('+')
    out = dict(gap=(100, 100), sp=None, stack4=False, features=False, noprox=False)
    for p in parts:
        if p.startswith('gap'): out['gap'] = _rng(p, 'gap')
        elif p.startswith('sp'): out['sp'] = _rng(p, 'sp')
        elif p in AGENT: out[p] = True
        else: raise ValueError(f'unknown spec part {p!r}')
    return out


def randomness_part(spec):
    return '+'.join(p for p in spec.split('+') if p.startswith(RANDOMNESS))


def agent_part(spec):
    return '+'.join(p for p in spec.split('+') if p in AGENT)


def with_randomness(model_spec, randomness):
    """Evaluation env: same agent parts as the model, different randomness."""
    a = agent_part(model_spec)
    return randomness + ('+' + a if a else '')


def make(spec):
    s = parse(spec)
    reward = _NoProximity if s['noprox'] else _OriginalProximity
    kw = dict(use_lidar=not s['features'])
    if s['sp'] is None:
        cls = type('Env', (reward, RandomGapEnv), {})
        env = cls(s['gap'], **kw)
    else:
        base = GapSpacingEnv if not s['noprox'] else type('Env', (_NoProximity, GapSpacingEnv), {})
        env = base(s['gap'], s['sp'], **kw)
    if s['stack4']:
        env = gym.wrappers.FrameStackObservation(env, 4)
    return env
