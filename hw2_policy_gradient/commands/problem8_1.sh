#!/bin/bash
set -eux

for b in 10000 30000 50000
do
	for r in 0.005 0.01 0.02
	do
		python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b ${b} -lr ${r} -rtg --nn_baseline --exp_name halfcheetah_b${b}_r${r}
	done
done

python plot.py data/halfcheetah_b10000*
python plot.py data/halfcheetah_b30000*
python plot.py data/halfcheetah_b50000*

