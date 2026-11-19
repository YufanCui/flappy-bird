#!/bin/zsh
# usage: job.sh <tag> <seed>
cd /Users/cynthia/flappy-bird/project
export PYTHONDONTWRITEBYTECODE=1
O=results/improve_r3_20261118
tag=$1; seed=$2
case $tag in
  _lr3M) extra=(--steps 3000000 --lr-linear) ;;
  _lr)   extra=(--steps 1000000 --lr-linear) ;;
  _base) extra=(--steps 1000000) ;;
  *) echo "unknown tag $tag"; exit 1 ;;
esac
/Users/cynthia/miniforge3/envs/rl/bin/python train_v3.py --out $O --spec gap80_120+stack4 --seed $seed --ckpt-every 100000 --tag $tag $extra > $O/logs/${tag}_seed$seed.log 2>&1 && echo "trained $tag $seed" || echo "FAILED $tag $seed"
