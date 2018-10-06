## Explanations, answers of the questions are in writeup.md ##

### Usage: ###

###	Problem 4: I used one 32-neuron fully connected layer, learning rate: 1e-2, iteration: 100, seed: 3. ###
Graph 1: comparisons of small batch size with/without rtg or na.
```shell
./problem4_sb.sh
```
Graph 2: comparisons of large batch size with/without rtg or na.
```shell
./problem4_lb.sh
```

###	Problem 5: I used two 64-neuron fully connected layers, max_iteration: 100, discount factor: 0.9, seed: 3. ###
I have different commands in this shell file, one to test batch size 1000 with different learning rates, one to test combinations of batch size 500-1000 (incremented by 100) and learning rate 5e-3-10e-3 (incremented by 1e-3). Also some small batch sizes. To run it, comment/uncomment parts of the shell file.
```shell
./problem5.sh
```

###	Problem 7: I used the provided command (seed=3) ###
```shell
./problem7.sh
```

###	Problem 8: I used the provided command###
```shell
./problem8_1.sh
./problem8_2.sh
```