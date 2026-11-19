"""Tests for env_v3 sticky actions and Exp B top penalty (2026-11-19). Run: python test_env_v3_sticky.py"""
import numpy as np
import env_v3


def scripted(env):
    u = env.unwrapped
    pipes = [p for p in u._lower_pipes if p['x'] + 52 >= u._player_x]
    target = min(pipes, key=lambda p: p['x'])['y'] - 30 if pipes else 250
    return int(u._player_y > target and u._player_vel_y >= 0)


def run(spec, seed, steps=3000, actions=None):
    env = env_v3.make(spec); env.reset(seed=seed)
    log = []
    for i in range(steps):
        a = scripted(env) if actions is None else actions[i]
        obs, r, t, tr, info = env.step(a)
        u = env.unwrapped
        log.append(dict(a=info.get('executed_action', a), r=r, y=u._player_y,
                        pipes=tuple((p['x'], p['y']) for p in u._upper_pipes), t=t))
        if t or tr:
            break
    return log, info


def test_sticky_rate_and_pipes():
    log, info = run('gap80_120+sticky25', seed=7)
    assert info['sticky_overrides'] > 0
    # replaying the executed actions in the non-sticky env gives identical pipes, rewards and positions
    plain, _ = run('gap80_120', seed=7, steps=len(log), actions=[x['a'] for x in log])
    assert [x['pipes'] for x in plain] == [x['pipes'] for x in log]
    assert [x['r'] for x in plain] == [x['r'] for x in log]
    assert [x['y'] for x in plain] == [x['y'] for x in log]


def test_sticky_rng_separate():
    """The sticky RNG must not consume the env's np_random (which generates pipes)."""
    a = env_v3.make('gap80_120+sticky25'); b = env_v3.make('gap80_120')
    a.reset(seed=11); b.reset(seed=11)
    for i in range(40):
        a.step(0); b.step(0)
    assert a.unwrapped.np_random.bit_generator.state == b.unwrapped.np_random.bit_generator.state
    assert a.unwrapped._gap_rng.bit_generator.state == b.unwrapped._gap_rng.bit_generator.state if hasattr(b.unwrapped, '_gap_rng') else True


def test_sticky_frequency():
    env = env_v3.StickyActions(env_v3.make('gap100_100'), 0.25); env.reset(seed=1)
    over = 0; n = 0
    for i in range(2000):
        want = i % 2  # alternate so every repeat is an override
        _, _, t, tr, info = env.step(want)
        n += 1; over = info['sticky_overrides']
        if t or tr:
            env.reset(seed=1000 + i); over_base = 0
            break
    assert 0.15 < over / n < 0.35, over / n


def test_sticky_deterministic_seed():
    a, _ = run('gap80_120+sticky10', seed=3); b, _ = run('gap80_120+sticky10', seed=3)
    assert a == b


def test_topB():
    acts = [1] * 200  # keep flapping: goes above the screen
    base, _ = run('gap100_100', seed=5, steps=200, actions=acts)
    topb, _ = run('gap100_100+topB', seed=5, steps=200, actions=acts)
    assert [x['y'] for x in base] == [x['y'] for x in topb]
    above = 0
    for x, y in zip(base, topb):
        if x['y'] < 0 and not x['t']:
            assert x['r'] == -0.5 and y['r'] == -2.0; above += 1
        else:
            assert x['r'] == y['r']
    assert above > 5


def test_prevact():
    env = env_v3.make('gap80_120+sticky25+prevact'); obs, _ = env.reset(seed=4)
    assert obs.shape == (181,) and obs[-1] == 0.0
    for i in range(60):
        obs, _, t, tr, info = env.step(i % 2)
        assert obs[-1] == info['executed_action']
        assert env.observation_space.contains(obs)
        if t or tr:
            break
    st = env_v3.make('gap80_120+sticky5+prevact+stack4'); o, _ = st.reset(seed=1)
    assert o.shape == (4, 181)


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f(); print('ok', name)
