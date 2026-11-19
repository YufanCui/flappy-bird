#!/bin/zsh
# usage: RUN=<run_dir> CONDS="gap100_100 ..." [MODES="deterministic stochastic"] eval_one.sh <label>
cd /Users/cynthia/flappy-bird/project
export PYTHONDONTWRITEBYTECODE=1
label=$1
/Users/cynthia/miniforge3/envs/rl/bin/python eval_v3.py --run $RUN --label "$label" --conditions ${=CONDS} --modes ${=MODES:-deterministic stochastic} > "$RUN/logs/eval_$label.log" 2>&1 && echo "evaluated $label" || echo "FAILED $label"
