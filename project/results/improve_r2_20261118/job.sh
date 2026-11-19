#!/bin/zsh
# usage: job.sh <tag> <seed>
cd /Users/cynthia/flappy-bird/project
export PYTHONDONTWRITEBYTECODE=1
O=results/improve_r2_20261118
tag=$1; seed=$2
case $tag in
  _3M)  extra=(--steps 3000000) ;;
  _lr)  extra=(--steps 1000000 --lr-linear) ;;
  _cur) extra=(--steps 1000000 --curriculum 0.5) ;;
  _ent) extra=(--steps 1000000 --ppo '{"ent_coef": 0.01}') ;;
  _big) extra=(--steps 1000000 --ppo '{"policy_kwargs": {"net_arch": [128, 128]}}') ;;
  *) echo "unknown tag $tag"; exit 1 ;;
esac
/Users/cynthia/miniforge3/envs/rl/bin/python train_v3.py --out $O --spec gap80_120+stack4 --seed $seed --ckpt-every 100000 --tag $tag $extra > $O/logs/${tag}_seed$seed.log 2>&1 && echo "trained $tag $seed" || echo "FAILED $tag $seed"
