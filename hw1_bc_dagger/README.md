Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

Look at `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.

Author: Zhe Zheng

### Configuartion
NN:
 * two layer FC, 128 and 64 neurons respectively (also tried different combinations of 64, 128, 256)
 * tanh activation, xavier initializer

CNN:
 * Conv_1, 128 filters, kernel_size=5, ReLU and Maxpooling
 * Conv_2, 64 filters, kernel_size=3, ReLU and Maxpooling
 * Flatten and output layer

Note:
 * I was able to train a pretty decent model for Humanoid-v2 using CNN and 10 iterations of Dagger
 * Intuitively, CNN does not make sense for this task, probably it is the expressiveness of CNN or the input data of Humanoid-v2 is actually
 coordinates of joints, so extract local information would make sense (for instance, extract information for knee movement, ankle movement separately)
 * I did not fine tune all the parameters.
 * Humanoid-v2 is the hardest model to train according to our TA.
[Result of CNN for Humanoid-v2]: https://github.com/Russzheng/CS294_Deep_Reinforcement_Learning/upload/master/hw1_bc_dagger/dagger.png "Logo Title Text 2"


### Example usage:
```shell
    python behavioral_cloning_DAGGER.py --envidx 0 --render --num_rollouts 20 --dagger
```

As for envidx, check EXPERTS = ['Hopper-v2', 'Ant-v2', 'HalfCheetah-v2', 'Humanoid-v2', 'Reacher-v2', 'Walker2d-v2']
the index corresponds to this list.

Also, as for CNN, need to reshape observations' shape to (.., .., 1). I already commented in the code (both data loading and model training parts).

In model_training, comment and uncomment different parts for different model setups (NN and CNN).

I have commented places that need to be changed when switching between CNN and NN model. Comment/Uncomment layers/reshaping depending on which model
you want to use.

