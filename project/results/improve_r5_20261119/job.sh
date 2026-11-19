#!/bin/zsh
# usage: job.sh <tag> <seed>
cd /Users/cynthia/flappy-bird/project
export PYTHONDONTWRITEBYTECODE=1
O=results/improve_r5_20261119
tag=$1; seed=$2
case $tag in
  _prev)     spec=gap80_120+sticky5+stack4+prevact; steps=3000000 ;;
  _10M)      spec=gap80_120+sticky5+stack4; steps=10000000 ;;
  _prev10M)  spec=gap80_120+sticky5+stack4+prevact; steps=10000000 ;;
  *) echo "unknown tag $tag"; exit 1 ;;
esac
/Users/cynthia/miniforge3/envs/rl/bin/python train_v3.py --out $O --spec $spec --seed $seed --steps $steps --lr-linear --ckpt-every 1000000 --tag $tag > $O/logs/${tag}_seed$seed.log 2>&1 && echo "trained $tag $seed" || echo "FAILED $tag $seed"
