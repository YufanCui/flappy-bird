"""Summarize diagnostics_v1 output: per-checkpoint table, failure taxonomy, per-pipe passages.

Definitions (fixed before looking at the summary):
- pipes_cleared: number of times the bird's left edge passes a pipe's right edge (next pipe changes).
  This is a full-body clearance count, distinct from the inherited midpoint game_score.
- Failure taxonomy for terminated episodes, by pipes_cleared at death and crash_type:
    first_pipe_upper / first_pipe_lower / first_pipe_ground  (pipes_cleared == 0)
    later_pipe_upper / later_pipe_lower / later_pipe_ground  (pipes_cleared >= 1)
  Truncated episodes (3,000-step cap) are counted separately.
- A passage = steps during which a given pipe is the next pipe and the bird clears it.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def load(path):
    return list(csv.DictReader(path.open()))


def main(run):
    run = Path(run)
    steps = load(run / 'steps.csv')
    eps = defaultdict(list)
    for s in steps:
        eps[(s['group'], s['train_seed'], s['condition'], s['episode_seed'])].append(s)
    episode_rows, passages = [], []
    for key, rows in eps.items():
        cleared, cur, cur_rows = 0, None, []
        for r in rows:
            x = float(r['next_pipe_x'])
            if cur is not None and x > cur:  # next pipe changed: previous pipe fully cleared
                passages.append(dict(group=key[0], train_seed=key[1], condition=key[2],
                    gap=float(cur_rows[0]['next_pipe_gap']), steps=len(cur_rows),
                    prox_pipe=sum(q['proximity_source'] == 'pipe' for q in cur_rows),
                    flaps=sum(int(q['action']) for q in cur_rows)))
                cleared += 1; cur_rows = []
            cur = x; cur_rows.append(r)
        last = rows[-1]
        if last['truncated'] == 'True':
            failure = 'truncated'
        else:
            failure = ('first_pipe_' if cleared == 0 else 'later_pipe_') + last['crash_type'].replace('_pipe', '')
        cases = Counter(r['reward_case'] for r in rows)
        episode_rows.append(dict(group=key[0], train_seed=key[1], condition=key[2], episode_seed=key[3],
            game_score=int(last['score']), pipes_cleared=cleared, steps=len(rows), failure=failure,
            ret=sum(float(r['reward']) for r in rows), flap_rate=np.mean([int(r['action']) for r in rows]),
            **{f'n_{c}': cases.get(c, 0) for c in ['alive', 'score', 'proximity', 'above_screen', 'crash']},
            prox_pipe=sum(r['proximity_source'] == 'pipe' for r in rows),
            prox_ground=sum(r['proximity_source'] == 'ground' for r in rows),
            death_offset=float(last['bird_center_offset']), death_gap=float(last['next_pipe_gap'])))

    # Per-checkpoint table.
    table = []
    by = defaultdict(list)
    for e in episode_rows:
        by[(e['group'], e['train_seed'], e['condition'])].append(e)
    for (g, s, c), es in sorted(by.items()):
        fail = Counter(e['failure'] for e in es)
        n = len(es)
        table.append(dict(group=g, train_seed=s, condition=c, episodes=n,
            mean_game_score=np.mean([e['game_score'] for e in es]),
            mean_pipes_cleared=np.mean([e['pipes_cleared'] for e in es]),
            pct_zero_cleared=100 * np.mean([e['pipes_cleared'] == 0 for e in es]),
            mean_steps=np.mean([e['steps'] for e in es]), mean_return=np.mean([e['ret'] for e in es]),
            flap_rate=np.mean([e['flap_rate'] for e in es]),
            prox_steps_per_ep=np.mean([e['n_proximity'] for e in es]),
            above_screen_per_ep=np.mean([e['n_above_screen'] for e in es]),
            prox_ground_per_ep=np.mean([e['prox_ground'] for e in es]),
            pct_upper=100 * sum(v for k, v in fail.items() if k.endswith('upper')) / n,
            pct_lower=100 * sum(v for k, v in fail.items() if k.endswith('lower')) / n,
            pct_ground=100 * sum(v for k, v in fail.items() if k.endswith('ground')) / n,
            pct_truncated=100 * fail.get('truncated', 0) / n,
            mean_death_offset=np.mean([e['death_offset'] for e in es if e['failure'] != 'truncated'])))
    taxonomy = Counter(e['failure'] for e in episode_rows)
    # Passages by gap (all policies pooled; descriptive only).
    pas = defaultdict(list)
    for p in passages:
        pas[int(p['gap'] // 10 * 10) if p['condition'] == 'random80_120' else 'fixed100'].append(p)
    passage_summary = {str(k): dict(n=len(v), mean_prox_pipe_steps=float(np.mean([p['prox_pipe'] for p in v])),
        pct_zero_prox=float(100 * np.mean([p['prox_pipe'] == 0 for p in v])),
        mean_passage_steps=float(np.mean([p['steps'] for p in v]))) for k, v in sorted(pas.items(), key=str)}
    # Return decomposition pooled over all episodes.
    tot = Counter()
    for e in episode_rows:
        for c, w in [('alive', .1), ('score', 1), ('proximity', -1), ('above_screen', -.5), ('crash', -1)]:
            tot[c] += w * e[f'n_{c}']
    out = dict(episodes=len(episode_rows), steps=len(steps), passages=len(passages),
        failure_taxonomy=dict(taxonomy), passages_by_gap=passage_summary,
        pooled_return_contribution={k: round(v, 3) for k, v in tot.items()})
    (run / 'summary.json').write_text(json.dumps(out, indent=2) + '\n')
    with (run / 'checkpoint_table.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader()
        w.writerows([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in table])
    with (run / 'episode_taxonomy.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(episode_rows[0])); w.writeheader(); w.writerows(episode_rows)
    with (run / 'passages.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(passages[0])); w.writeheader(); w.writerows(passages)
    print(json.dumps(out, indent=2))
    cols = ['group', 'train_seed', 'condition', 'mean_game_score', 'mean_pipes_cleared', 'pct_zero_cleared',
            'mean_steps', 'mean_return', 'flap_rate', 'prox_steps_per_ep', 'above_screen_per_ep',
            'pct_upper', 'pct_lower', 'pct_ground', 'pct_truncated', 'mean_death_offset']
    print(' | '.join(cols))
    for r in table:
        print(' | '.join(f'{r[c]:.2f}' if isinstance(r[c], float) else str(r[c]) for c in cols))


if __name__ == '__main__':
    main(sys.argv[1])
