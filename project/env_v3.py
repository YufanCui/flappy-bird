"""Environment factory v3 (2026-11-18). Spec string = '+'-joined parts, e.g. 'gap80_120+stack4'.

Randomness parts (what the environment samples):
  gapA_B     vertical gap size G ~ U{A..B} per new pair (main randomness; gap100_100 = original game)
  spA_B      horizontal spacing D ~ U{A..B} per new pair (extra randomness; default original 144/144/164)
  stickyNN   sticky actions (added 2026-11-19): at every step, with probability NN/100 the environment ignores the
             agent's chosen action and repeats the previously executed action (Machado et al. 2018 style). Models
             human/hardware input imprecision; game rules and pipes are unchanged. Separate seeded RNG stream.
Agent/training parts (what the agent sees / is rewarded for):
  stack4     stack last 4 LiDAR observations (velocity information)
  features   vendor 12-dim feature observation (privileged; has no proximity penalty by vendor design)
  noprox     remove the proximity penalty (reward change only)
  prevact    append the previously EXECUTED action (0/1) to the LiDAR observation (added 2026-11-19). Under sticky
             actions this is what the next step may repeat. Information a human player also has (they saw the bird).
  topB       June Experiment B reward: above-screen penalty -0.5 -> -2 (crash penalty -1 still overrides it).
             Re-implemented from the documented change; the original June Exp B code was not recovered.
Default reward: original flappy-bird-gymnasium 0.4.0 (proximity -0.5). Vendor files are never modified.
"""
import gymnasium as gym
import numpy as np
from original_reward_env import _OriginalProximity
from random_gap_env import RandomGapEnv
from gap_spacing_env import GapSpacingEnv
from flappy_bird_gymnasium import FlappyBirdEnv

RANDOMNESS = ('gap', 'sp', 'sticky')
AGENT = ('stack4', 'features', 'noprox', 'topB', 'prevact')


class _NoProximity:
    def _get_observation_lidar(self):
        distances, _ = FlappyBirdEnv._get_observation_lidar(self)
        return distances, None


class StickyActions(gym.Wrapper):
    """With probability p the previously executed action is repeated instead of the chosen one.
    Own RNG stream: SeedSequence([reset seed, 0x535449]) so pipe sequences are identical to the non-sticky env."""
    def __init__(self, env, p):
        super().__init__(env); self.p = p; self.prev = 0
        self.rng = np.random.default_rng()
        self.repeats = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(np.random.SeedSequence([seed, 0x535449]))
        self.prev = 0; self.repeats = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        a = int(action)
        if self.rng.random() < self.p:
            if a != self.prev:
                self.repeats += 1
            a = self.prev
        self.prev = a
        obs, r, t, tr, info = self.env.step(a)
        info['executed_action'] = a; info['sticky_overrides'] = self.repeats
        return obs, r, t, tr, info


class TopPenaltyB(gym.Wrapper):
    """Exp B: when the bird is above the screen and did not crash this step, reward -2 instead of -0.5."""
    def step(self, action):
        obs, r, t, tr, info = self.env.step(action)
        if not t and self.env.unwrapped._player_y < 0:
            r = -2.0
        return obs, r, t, tr, info


class PrevAction(gym.Wrapper):
    """Observation = [LiDAR..., previously executed action]. Uses info['executed_action'] when sticky is active."""
    def __init__(self, env):
        super().__init__(env)
        lo, hi = env.observation_space.low, env.observation_space.high
        self.observation_space = gym.spaces.Box(np.append(lo, 0.0).astype(np.float64), np.append(hi, 1.0).astype(np.float64),
                                                dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return np.append(obs, 0.0), info

    def step(self, action):
        obs, r, t, tr, info = self.env.step(action)
        return np.append(obs, float(info.get('executed_action', int(action)))), r, t, tr, info


def _rng(part, prefix):
    a, b = part[len(prefix):].split('_')
    return int(a), int(b)


def parse(spec):
    parts = spec.split('+')
    out = dict(gap=(100, 100), sp=None, sticky=0.0, stack4=False, features=False, noprox=False, topB=False, prevact=False)
    for p in parts:
        if p.startswith('gap'): out['gap'] = _rng(p, 'gap')
        elif p.startswith('sticky'): out['sticky'] = int(p[len('sticky'):]) / 100
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
    if s['topB']:
        env = TopPenaltyB(env)
    if s['sticky'] > 0:
        env = StickyActions(env, s['sticky'])
    if s['prevact']:
        env = PrevAction(env)
    if s['stack4']:
        env = gym.wrappers.FrameStackObservation(env, 4)
    return env
