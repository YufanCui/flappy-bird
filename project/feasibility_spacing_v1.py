"""Exact reachability: can ANY action sequence survive two consecutive pipes at spacing D?

Worst case for horizontal spacing is a large vertical change between consecutive gap centers.
We place pipe 1 at x=288 with center c1 and pipe 2 at x=288+D with center c2 (gap 100), then
enumerate every reachable bird state (player_y, vel_y, flapped) step by step with the real env
(deepcopy), dropping crashed branches. Feasible = some branch fully clears pipe 2
(bird left edge past pipe 2's right edge). Rotation only affects LiDAR (reward), not dynamics,
so it is excluded from the state key; this is exact for survival. No policy or learning involved.
"""
import argparse
import copy
import json
from pathlib import Path
from random_spacing_env import RandomSpacingEnv
from flappy_bird_gymnasium.envs.constants import PIPE_HEIGHT, PIPE_WIDTH


def setup(D, c1, c2):
    e = RandomSpacingEnv((D, D)); e.reset(seed=0)
    for i, c in enumerate([c1, c2, c2]):
        top = c - 50
        e._upper_pipes[i]['y'] = top - PIPE_HEIGHT
        e._lower_pipes[i]['y'] = top + 100
    return e


def feasible(D, c1, c2, max_steps=400):
    e = setup(D, c1, c2)
    target_x = e._upper_pipes[1]['x']
    frontier = {(e._player_y, e._player_vel_y, e._player_flapped): e}
    for step in range(1, max_steps + 1):
        nxt = {}
        for env in frontier.values():
            for a in (0, 1):
                c = copy.deepcopy(env)
                _, _, t, _, _ = c.step(a)
                if t:
                    continue
                if c._player_x > target_x - 4 * step + PIPE_WIDTH:
                    return True, step
                nxt.setdefault((c._player_y, c._player_vel_y, c._player_flapped), c)
        if not nxt:
            return False, step
        frontier = nxt
    return False, max_steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--spacings', type=int, nargs='+', default=[115, 120, 125, 130, 140, 150])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for D in a.spacings:
        for c1, c2 in [(220, 150), (150, 220)]:  # extreme center jumps: low->high and high->low
            ok, steps = feasible(D, c1, c2)
            rows.append(dict(spacing=D, center_from=c1, center_to=c2, feasible=ok, steps=steps))
            print(json.dumps(rows[-1]), flush=True)
    (a.out / 'spacing_feasibility.json').write_text(json.dumps(dict(
        note='Exact survival reachability through two pipes with extreme center change (70 px), gap 100.',
        rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
