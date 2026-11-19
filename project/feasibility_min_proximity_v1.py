"""Exact search: minimum number of proximity-penalty steps needed to fully clear the first pipe.

Pipe motion and generation do not depend on actions, so for a given episode seed the only
action-dependent state is the bird's (player_y, vel_y, rot, flapped). We expand every reachable
state with both actions step by step using the real environment (deepcopy), deduplicate identical
bird states keeping the fewest penalties, and stop when the bird's left edge passes the first
pipe's right edge. No policy or learning is involved.

Cost counted = steps whose LiDAR min distance < PLAYER_PRIVATE_ZONE (the inherited -1 case),
excluding the step that scores (score reward overrides proximity). Crashing branches are dropped.
"""
import argparse
import copy
import json
from pathlib import Path
from random_gap_env import RandomGapEnv
from flappy_bird_gymnasium.envs.constants import PIPE_WIDTH


def search(gap, seed, max_steps=200):
    env = RandomGapEnv((gap, gap))
    env.reset(seed=seed)
    first_x = min(p['x'] for p in env.pipe_geometry())
    target = first_x  # track this pipe by its x (all pipes move together)
    frontier = {(): (0, env)}
    for step in range(1, max_steps + 1):
        nxt = {}
        for _, (cost, e) in frontier.items():
            for action in (0, 1):
                c = copy.deepcopy(e)
                _, r, t, _, info = c.step(action)
                if t:
                    continue
                pipe = min(c.pipe_geometry(), key=lambda p: abs(p['x'] - (target + 4 * -step)))
                k = (c._player_y, c._player_vel_y, c._player_rot, c._player_flapped)
                new = cost + (r == -1)
                if c._player_x > pipe['x'] + PIPE_WIDTH:  # left edge past right edge: cleared
                    return dict(gap=gap, seed=seed, min_proximity_steps=new, steps=step,
                                cleared=True, states_last=len(frontier))
                if k not in nxt or nxt[k][0] > new:
                    nxt[k] = (new, c)
        if not nxt:
            return dict(gap=gap, seed=seed, cleared=False, steps=step)
        best = min(v[0] for v in nxt.values())
        frontier = {k: v for k, v in nxt.items() if v[0] <= best + 3}
    return dict(gap=gap, seed=seed, cleared=False, steps=max_steps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gaps', type=int, nargs='+', default=[80, 90, 100, 110, 120])
    p.add_argument('--seeds', type=int, nargs='+', default=[220000, 220001, 220002])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for g in a.gaps:
        for s in a.seeds:
            rows.append(search(g, s)); print(json.dumps(rows[-1]), flush=True)
    (a.out / 'feasibility.json').write_text(json.dumps(dict(
        note='Minimum proximity steps to clear first pipe, exact over pruned (best+3) state set.',
        rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
