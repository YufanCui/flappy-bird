"""Summarize eval_v3 outputs: seed-level means of pipes_cleared, all seeds listed, perfect-episode rate.

Usage: python summarize_v3.py results/gap_main_v3_20261118 [label_filter_substring]
A 'perfect' episode = reached the 10,000-step cap without crashing (truncated), i.e. >= ~270 pipes.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np


def main(run, filt=''):
    run = Path(run)
    rows = [r for f in sorted((run / 'eval').glob('*.csv')) if filt in f.name for r in csv.DictReader(f.open())]
    per = defaultdict(list)
    for r in rows:
        group = r['label'].rsplit('_seed', 1)[0]
        per[(group, r['checkpoint'], r['condition'], r['mode'], r['train_seed'])].append(r)
    agg = defaultdict(list)
    for (g, ck, c, m, s), rs in per.items():
        cl = [int(x['pipes_cleared']) for x in rs]
        agg[(g, ck, c, m)].append(dict(seed=int(s), mean=float(np.mean(cl)), median=float(np.median(cl)), n=len(rs),
                                       perfect=float(np.mean([x['truncated'] == 'True' for x in rs])),
                                       max=max(cl)))
    table = []
    for (g, ck, c, m), v in sorted(agg.items()):
        v = sorted(v, key=lambda x: x['seed'])
        means = [x['mean'] for x in v]
        table.append(dict(group=g, checkpoint=ck, condition=c, mode=m, n_seeds=len(v),
            episodes_per_seed=sorted({x['n'] for x in v}), mean=round(float(np.mean(means)), 2),
            sd=round(float(np.std(means, ddof=1)), 2) if len(means) > 1 else None,
            seed_means=[round(x, 1) for x in means], perfect_rate=round(float(np.mean([x['perfect'] for x in v])), 3),
            best_episode=max(x['max'] for x in v)))
    tag = filt or 'all'
    (run / f'summary_{tag}.json').write_text(json.dumps(table, indent=2) + '\n')
    for t in table:
        print(f"{t['group']:28s} {t['checkpoint']:6s} {t['condition']:22s} {t['mode'][:5]:5s} "
              f"mean={t['mean']:7.2f} sd={t['sd']} perfect={t['perfect_rate']:.2f} best={t['best_episode']} seeds={t['seed_means']}")


if __name__ == '__main__':
    main(*sys.argv[1:])
