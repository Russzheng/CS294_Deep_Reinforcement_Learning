"""

Behavorial cloning and DAGGER algorithm, CS294 hw 1
Author: Zhe Zheng

Example usage:
    python behavioral_cloning_DAGGER.py --envidx 0 --render --num_rollouts 20 --dagger

    As for envidx, check EXPERTS = ['Hopper-v2', 'Ant-v2', 'HalfCheetah-v2', 'Humanoid-v2', 'Reacher-v2', 'Walker2d-v2']
    the index corresponds to this list.

Also, as for CNN, need to reshape observations' shape to (.., .., 1)

In model_training, comment and uncomment different lines for different model setups (NN and CNN).

In main function, comment or uncomment to use BC alone or Dagger or use the dagger parameter


"""

import pickle
import gym
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Convolution1D, MaxPooling1D, Activation, Flatten, BatchNormalization
from keras.utils import np_utils
import tf_util
import load_policy
import argparse
import time

#### Parse agruments ####
parser = argparse.ArgumentParser()
parser.add_argument('--envidx', type=int)
parser.add_argument('--render', action='store_true')
parser.add_argument('--num_rollouts', type=int, default=20,
                    help='Number of expert roll outs')
parser.add_argument("--max_timesteps", type=int)
parser.add_argument('--dagger', action='store_true')
args = parser.parse_args()

#### For debugging usage. ####
# import sys; print(sys._getframe().f_code.co_name,sys._getframe().f_lineno)
# from IPython import embed; embed()

# List of experts
EXPERTS = ['Hopper-v2', 'Ant-v2', 'HalfCheetah-v2', 'Humanoid-v2', 'Reacher-v2', 'Walker2d-v2']

# some parameters input
idx_expert = args.envidx
num_roll_out = args.num_rollouts

# some hyperparameters
batch_size = 64
epoch = 20
split_ratio = 0.2
iter_dagger = 10

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def load_data(data):
    if data is None:
        with open('expert_data/' + EXPERTS[idx_expert] + '_' + str(num_roll_out) + '.pkl', 'rb') as f:
            data = pickle.load(f)

    X = data['observations']
    y = data['actions']
    y = y.reshape(y.shape[0], -1)
    # some basic dataset info
    print('Observations shape is', X.shape, "\nActions shape is", y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=0)

    # normalization
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)

    return X_train, X_test, y_train, y_test, data

def model_training(X_train, X_test, y_train, y_test, dagger):
    # Define our model
    model = Sequential([
        Convolution1D(nb_filter=128, kernel_size=5, input_shape=(X_train.shape[1], 1), activation='relu'),
        MaxPooling1D(),
        Convolution1D(nb_filter=64, kernel_size=3, activation='relu'),
        MaxPooling1D(),
        Flatten(),
        # Dense(256, kernel_initializer='glorot_normal', activation='tanh'),
        # Dense(128, kernel_initializer='glorot_normal', activation='tanh'),
        Dense(128, kernel_initializer='glorot_normal', activation='tanh'),
        # Dense(64,  kernel_initializer='glorot_normal', activation='tanh'),
        # Flatten(),
        Dense(y_train.shape[1], activation='linear'),
    ])

    # Train, evaluate and save our model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    time_callback = TimeHistory()
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch, callbacks=[time_callback])
    times = time_callback.times
    print('#### Average training time for one epoch is:', np.mean(np.array(times)), '####')

    score = model.evaluate(X_test, y_test)
    if dagger:
        model.save('models/' + EXPERTS[idx_expert]+ '_rollout_' + str(num_roll_out) + '_dagger_model.h5')
    else:
        model.save('models/' + EXPERTS[idx_expert]+ '_rollout_' + str(num_roll_out) + '_bc_model.h5')

def load_model_comparison():
    # With reference to run_experts.py obviously

    with tf.Session():
        tf_util.initialize()
        env = gym.make(EXPERTS[idx_expert])
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        new_observations = []

        model = load_model('models/' + EXPERTS[idx_expert] + '_rollout_' + str(num_roll_out) + '_bc_model.h5')
        for i in range(num_roll_out):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                obs = np.array(obs)
                # obs = obs.reshape(1, len(obs))
                # CNN
                obs = obs.reshape(1, len(obs), 1)
                action = (model.predict(obs, batch_size=batch_size, verbose=0))
                new_observations.append(obs)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if (args.render):
                    env.render()
                if steps % 1000 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

def dagger():
    # With reference to run_experts.py
    
    # expert policy
    policy_fn = load_policy.load_policy('experts/' + EXPERTS[idx_expert] + '.pkl')
    data = None
    all_returns = []
    all_stds = []

    for iteration in range(iter_dagger):
        # Train on given dataset first
        X_train, X_test, y_train, y_test, data = load_data(data)

        # Reshaping
        # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
        # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
        # CNN
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        model_training(X_train, X_test, y_train, y_test, True)

        with tf.Session():
            tf_util.initialize()
            env = gym.make(EXPERTS[idx_expert])
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            new_observations = []
            new_expert_actions = []

            model = load_model('models/' + EXPERTS[idx_expert] + '_rollout_' + str(num_roll_out) + '_dagger_model.h5')
            for i in range(num_roll_out):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    # run the expert policy
                    expert_action = policy_fn(obs[None,:])
                    new_expert_actions.append(expert_action)

                    # running our model
                    obs = np.array(obs)
                    # obs = obs.reshape(1, len(obs))
                    # for cnn
                    obs = obs.reshape(1, len(obs), 1)
                    action = (model.predict(obs, batch_size=batch_size, verbose=0))
                    new_observations.append(obs)

                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if (args.render):
                        env.render()
                    if steps % 1000 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
            all_returns.append(np.mean(returns))
            all_stds.append(np.std(returns))

            # dataset aggregation
            new_observations = np.array(new_observations)
            new_observations = new_observations.reshape(new_observations.shape[0], data['observations'].shape[1]) 
            new_expert_actions = np.array(new_expert_actions)
            new_expert_actions = new_expert_actions.reshape(new_expert_actions.shape[0], data['actions'].shape[1], -1)
            
            data['observations'] = np.concatenate((data['observations'], new_observations), axis=0)  
            data['actions'] = np.concatenate((data['actions'], new_expert_actions), axis=0)  

    print('All returns generated by dagger', all_returns)
    print('All stds generated by dagger', all_stds)

def main():
    if not args.dagger:
        ##### FOR BC ALONE START #####
        #### Load data, and split into training and validation set ####
        X_train, X_test, y_train, y_test, _ = load_data(None)

        #### Reshaping ####
        # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
        # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
        # CNN
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        #### Define our model ####
        model_training(X_train, X_test, y_train, y_test, False)

        ## Load models and performance comparison ####
        load_model_comparison()

        ##### FOR BC ALONE END #####

    if args.dagger:
        ##### DAGGER START #####
        dagger()
        ##### DAGGER END #####

if __name__ == '__main__':
    main()
