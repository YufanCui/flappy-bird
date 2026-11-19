#!/bin/zsh
cd /Users/cynthia/flappy-bird/project
export PYTHONDONTWRITEBYTECODE=1
cat results/gap_main_v3_20261118/jobs.txt | xargs -P 9 -n 5 sh -c 'tag=$4; [ "$tag" = none ] && tag=""; /Users/cynthia/miniforge3/envs/rl/bin/python train_v3.py --out results/gap_main_v3_20261118 --spec $0 --seed $1 --steps $2 --ckpt-every $3 --tag "$tag" > results/gap_main_v3_20261118/logs/$0${tag}_seed$1.log 2>&1 && echo "trained $0$tag $1"'
