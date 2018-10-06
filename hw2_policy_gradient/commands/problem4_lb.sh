#!/bin/bash
set -eux

rm -rf data/lb_*
# python train_pg_f18.py CartPole-v0 -l 1 -s 32 -lr 1e-2 -n 100 -b 1000 -e 3 -rtg --exp_name lb_sb_rtg_na_small
python train_pg_f18.py CartPole-v0 -l 1 -s 32 -lr 1e-2 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na
# python train_pg_f18.py CartPole-v0 -l 1 -s 32 -lr 1e-2 -n 100 -b 1000 -e 3 --exp_name lb_sb_no_rtg_na_small
python train_pg_f18.py CartPole-v0 -l 1 -s 32 -lr 1e-2 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna
# python train_pg_f18.py CartPole-v0 -l 1 -s 32 -lr 1e-2 -n 100 -b 1000 -e 3 -dna -rtg --exp_name lb_sb_rtg_dna_small
python train_pg_f18.py CartPole-v0 -l 1 -s 32 -lr 1e-2 -n 100 -b 5000 -e 3 -dna -rtg --exp_name lb_rtg_dna

python plot.py data/lb_*