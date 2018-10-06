#!/bin/bash
set -eux

rm -rf data/sb_*
python train_pg_f18.py CartPole-v0 -l 1 -s 32 -lr 1e-2 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na
python train_pg_f18.py CartPole-v0 -l 1 -s 32 -lr 1e-2 -n 100 -b 1000 -e 3 --exp_name sb_no_rtg_na
python train_pg_f18.py CartPole-v0 -l 1 -s 32 -lr 1e-2 -n 100 -b 1000 -e 3 -dna -rtg --exp_name sb_rtg_dna
python train_pg_f18.py CartPole-v0 -l 1 -s 32 -lr 1e-2 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna

python plot.py data/sb_*