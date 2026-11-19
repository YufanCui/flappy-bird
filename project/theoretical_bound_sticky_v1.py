"""Theoretical upper bound on performance under sticky actions (2026-11-19).

Question: with sticky actions (probability p the previous executed action is repeated), how far can ANY policy fly?

Method (exact dynamic programming, no learning):
- For a given episode seed, the pipe sequence does not depend on the bird's actions (pipes move 4 px/step and
  respawn with heights from the env's own RNG streams). We record the exact pipe rectangles for every step by
  stepping the real env (env_v3.make(spec_without_sticky)).
- Bird state before a step = (y, vel_y, previously executed action). Dynamics and collision copy the vendor code
  (flap only if y > -2*PLAYER_HEIGHT; gravity +1 up to 10 when not flapping; y += min(vel, ground - y - h);
  pygame.Rect integer truncation; ground crash if y + h >= ground_y - 1). Rotation does not affect collision.
- The "clairvoyant" optimal policy knows the whole future pipe sequence (more information than the agent, whose
  LiDAR cannot see pipes before they are generated) but not the future sticky coin flips. Backward induction gives
  S_k = max probability of clearing at least k pipes. Hence for ANY policy, P(pipes >= k) <= S_k and
  E[pipes] <= sum_k S_k. This is an UPPER BOUND, not an achievable target.
- Validation (--validate): (1) the vectorized crash test equals env._check_crash on sampled states; (2) with p=0 the
  DP survival probability over the full horizon is 1 (consistent with the earlier exact feasibility search); (3) the simulator's bird trajectory equals
  the real env's for random action sequences.

Usage: python theoretical_bound_sticky_v1.py --out results/theory_sticky_v1_<date> [--validate]
"""
import argparse
import json
from pathlib import Path
import numpy as np
import env_v3
from eval_spacing_pilot import PIPE_WIDTH
from flappy_bird_gymnasium.envs.constants import (PIPE_HEIGHT, PLAYER_HEIGHT, PLAYER_WIDTH, PLAYER_ACC_Y,
                                                  PLAYER_FLAP_ACC, PLAYER_MAX_VEL_Y)

Y0, Y1 = -120, 420          # bird y range covered by the state grid
VELS = np.arange(-9, PLAYER_MAX_VEL_Y + 1)
YS = np.arange(Y0, Y1)


def record_pipes(spec, seed, steps):
    """Pipe rects after each step t=1..steps (what _check_crash sees), and cumulative pipes cleared."""
    env = env_v3.make(spec); env.reset(seed=seed); u = env.unwrapped
    px = u._player_x; ground = u._ground['y']
    rects = np.zeros((steps + 1, 3, 3), dtype=np.int64)  # (x, upper_bottom_exclusive, lower_top)
    cleared = np.zeros(steps + 1, dtype=np.int64); n = 0
    for t in range(1, steps + 1):
        before = [p['x'] for p in u._upper_pipes]
        u._player_y = 200; u._player_vel_y = 0  # keep the bird alive; pipe motion is independent of the bird
        env.step(0)
        for b, p in zip(before, u._upper_pipes):
            if p['x'] < b and b + PIPE_WIDTH >= px > p['x'] + PIPE_WIDTH:
                n += 1
        cleared[t] = n
        for i, (up, lo) in enumerate(zip(u._upper_pipes, u._lower_pipes)):
            rects[t, i] = (int(up['x']), int(up['y']) + PIPE_HEIGHT, int(lo['y']))
    return rects, cleared, px, ground, (u, env)


def alive_mask(rects_t, px, ground):
    """alive[y] for bird at (px, y) against the 3 pipes of one step (pygame colliderect semantics)."""
    y = YS
    ok = ~(y + PLAYER_HEIGHT >= ground - 1)
    for x, up_bottom, lo_top in rects_t:
        if x < px + PLAYER_WIDTH and px < x + PIPE_WIDTH:          # x-overlap
            # upper rect y-range [up_bottom-320, up_bottom), lower rect [lo_top, lo_top+320)
            hit_up = (y < up_bottom) & (up_bottom - PIPE_HEIGHT < y + PLAYER_HEIGHT)
            hit_lo = (y < lo_top + PIPE_HEIGHT) & (lo_top < y + PLAYER_HEIGHT)
            ok &= ~(hit_up | hit_lo)
    return ok


def transitions(ground):
    """next (y index, vel index, valid) for executed action e in {0,1}, for every (y, vel)."""
    yy, vv = np.meshgrid(YS, VELS, indexing='ij')
    out = []
    for e in (0, 1):
        flap = (e == 1) & (yy > -2 * PLAYER_HEIGHT)
        v = np.where(flap, PLAYER_FLAP_ACC, vv)
        v = np.where(~flap & (v < PLAYER_MAX_VEL_Y), v + PLAYER_ACC_Y, v)
        ny = yy + np.minimum(v, np.floor(ground - yy - PLAYER_HEIGHT)).astype(np.int64)
        # vendor: y += min(vel, ground_y - y - h) with float ground; y stays float only when the ground clamp binds,
        # which is a crash anyway (y + h >= ground - 1), so integer y is exact for all surviving states.
        valid = (ny >= Y0) & (ny < Y1)
        out.append((np.clip(ny - Y0, 0, len(YS) - 1), v - VELS[0], valid))
    return out


def survival_curve(rects, cleared, px, ground, p, ks, y0, v0):
    """S_k for each k in ks: max over policies of P(clear >= k pipes), starting from (y0, v0, prev=0)."""
    trans = transitions(ground)
    res = {}
    for k in ks:
        if cleared[-1] < k:
            res[k] = None; continue
        T = int(np.argmax(cleared >= k))       # first step at which k pipes are cleared
        V = np.ones((len(YS), len(VELS), 2))   # V[y, v, prev]
        for t in range(T, 0, -1):
            alive = alive_mask(rects[t], px, ground)
            nxt = []
            for e in (0, 1):
                ny, nv, valid = trans[e]
                nxt.append(np.where(valid & alive[ny], V[ny, nv, e], 0.0))
            # chosen a; executed = a w.p. 1-p, prev w.p. p
            newV = np.empty_like(V)
            for prev in (0, 1):
                best = np.maximum((1 - p) * nxt[0] + p * nxt[prev], (1 - p) * nxt[1] + p * nxt[prev])
                newV[:, :, prev] = best
            V = newV
        res[k] = float(V[y0 - Y0, v0 - VELS[0], 0])
    return res


def validate(spec):
    import copy
    rng = np.random.default_rng(0)
    rects, cleared, px, ground, (u, env) = record_pipes(spec, 230000, 600)
    # (3) simulator trajectory vs real env, and (1) crash test vs env._check_crash
    trans = transitions(ground)
    mism = 0; checks = 0
    for trial in range(30):
        e = env_v3.make(spec); e.reset(seed=230000); uu = e.unwrapped
        y, v = uu._player_y, uu._player_vel_y
        for t in range(1, 600):
            a = int(rng.random() < 0.12)
            _, _, term, _, _ = e.step(a)
            ny, nv, valid = trans[a]
            yi, vi = y - Y0, v - VELS[0]
            y, v = int(ny[yi, vi] + Y0), int(nv[yi, vi] + VELS[0])
            alive = alive_mask(rects[t], px, ground)[y - Y0]
            checks += 1
            if term:
                mism += alive  # sim says alive but env crashed
                break
            mism += (y != uu._player_y) or (v != uu._player_vel_y) or (not alive)
    # direct crash-mask check at sampled ys
    e = env_v3.make(spec); e.reset(seed=230000); uu = e.unwrapped
    cm = 0
    for t in range(1, 400):
        uu._player_y = 200; uu._player_vel_y = 0; e.step(0)
        mask = alive_mask(rects[t], px, ground)
        for yv in range(-60, 400, 7):
            uu._player_y = yv
            cm += (not uu._check_crash()) != bool(mask[yv - Y0])
    # (2) p = 0: the optimal policy must survive the whole horizon (geometric feasibility)
    S = survival_curve(rects, cleared, px, ground, 0.0, [int(cleared[-1])], 244, -9)
    return dict(trajectory_and_alive_mismatches=int(mism), trajectory_checks=checks, crash_mask_mismatches=int(cm),
                p0_survival_full_horizon=S)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--gaps', nargs='+', default=['gap80_120', 'gap100_100'])
    ap.add_argument('--ps', type=float, nargs='+', default=[0.0, 0.01, 0.02, 0.05, 0.10])
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 2, 5, 10, 20, 50, 100, 200])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.validate:
        v = validate('gap80_120'); print(json.dumps(v)); (a.out / 'validation.json').write_text(json.dumps(v, indent=2) + '\n')
    rows = []
    for gap in a.gaps:
        for s in range(230000, 230000 + a.seeds):
            steps = int(a.ks[-1] * 38.5) + 400
            rects, cleared, px, ground, _ = record_pipes(gap, s, steps)
            for p in a.ps:
                S = survival_curve(rects, cleared, px, ground, p, a.ks, 244, -9)
                rows.append(dict(gap=gap, episode_seed=s, p=p, S=S))
                print(json.dumps(rows[-1]), flush=True)
    summary = []
    for gap in a.gaps:
        for p in a.ps:
            rs = [r for r in rows if r['gap'] == gap and r['p'] == p]
            Smean = {k: float(np.mean([r['S'][k] for r in rs])) for k in a.ks}
            # per-pipe hazard from the last checkpoint: S_K ~ (1-h)^K
            K = a.ks[-1]; h = 1 - Smean[K] ** (1 / K) if Smean[K] > 0 else 1.0
            ub_trunc = sum((a.ks[i + 1] - a.ks[i]) * Smean[a.ks[i]] for i in range(len(a.ks) - 1)) + a.ks[0]
            summary.append(dict(gap=gap, p=p, mean_S=Smean, hazard_per_pipe_from_SK=h,
                                implied_mean_pipes_if_constant_hazard=(1 / h if h > 0 else None),
                                upper_bound_E_pipes_up_to_K=ub_trunc))
            print(json.dumps(summary[-1]), flush=True)
    (a.out / 'bound_rows.json').write_text(json.dumps(rows, indent=1) + '\n')
    (a.out / 'bound_summary.json').write_text(json.dumps(dict(
        note=__doc__.split('Usage')[0], summary=summary), indent=2) + '\n')


if __name__ == '__main__':
    main()
