#!/bin/zsh
# usage: eval_one.sh <run> <label>   (cap 40,000 steps)
cd /Users/cynthia/flappy-bird/project; export PYTHONDONTWRITEBYTECODE=1
/Users/cynthia/miniforge3/envs/rl/bin/python eval_v3.py --run $1 --label "$2" --cap 40000 --tag _cap40k --conditions gap80_120+sticky5 gap80_120 gap100_100 gap100_100+sticky5 > "$1/logs/eval_$2_cap40k.log" 2>&1 && echo "evaluated $2" || echo "FAILED $2"
