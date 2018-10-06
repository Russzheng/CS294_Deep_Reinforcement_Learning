#!/bin/bash
set -eux

#### First attempt ####
# for i in 1e-2 1e-3
# do
#  	python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b 1000 -lr ${i} -rtg --exp_name IP_b1000_r${i}
# done

#### Narrow down learning rate for batch_size=1000 ####
# for i in 5 6 7 8 9
# do
# 	python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b 1000 -lr ${i}e-3 -rtg --exp_name IP_b1000_r${i}e-3
# done

#### Try more combinations ####
for i in 5 6 7 8 9 10
do
	for j in 500 600 700 800 900
	do
		python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b ${j} -lr ${i}e-3 -rtg --exp_name IP_b${j}_r${i}e-3
	done
done

#### Plotting ####
# python plot.py data/IP_b1000*
# python plot.py data/IP_b500*
# python plot.py data/IP_b600*
# python plot.py data/IP_b700*
# python plot.py data/IP_b800*
# python plot.py data/IP_b900*
python plot.py data/IP_b500_r10e-3* data/IP_b1000_r6e-3* data/IP_b800_r9e-3* 

for b in 32 64 128 256 200 300
do
	python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b ${b} -lr 1e-2 -rtg --exp_name IP_b${b}_r1e-2
done

python plot.py data/IP_b256* data/IP_b200* data/IP_b300* 