import numpy as np
from gym import spaces
from gym import Env


class PointEnv(Env):
    """
    point mass on a 2-D plane
    goals are sampled randomly from a square
    """

    def __init__(self, granularity, num_tasks=1):
        self.granularity = granularity
        self.reset_task()
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, is_evaluation=False, train_test=True):
        '''
        sample a new task randomly

        Problem 3: make training and evaluation goals disjoint sets
        if `is_evaluation` is true, sample from the evaluation set,
        otherwise sample from the training set
        '''
        #====================================================================================#
        #                           ----------PROBLEM 3----------
        #====================================================================================#
        # YOUR CODE HERE
        if train_test:
            # make sure parameters are passed correctly
            # print('Granularity is:', self.granularity)
            '''
            granularity = 2 for 4x4
            1100
            1100
            0011
            0011
            Train on 1s and test on 0s.
            '''

            # if granularity = 2, 10 squares on each row/column
            # we choose square with row_num = i and col_num = j
            square_num = 20 / self.granularity
            row = np.random.randint(square_num)
            col = np.random.randint(square_num / 2)

            if is_evaluation:
                # choose from zeros
                x = self.granularity * row + np.random.uniform(0, self.granularity) - 10
                if row % 2 == 1:
                    y =  self.granularity * col * 2 + np.random.uniform(self.granularity, self.granularity * 2) - 10
                if row % 2 == 0:
                    y =  self.granularity * col * 2 + np.random.uniform(0, self.granularity) - 10
            else:
                # choose from ones
                x = self.granularity * row + np.random.uniform(0, self.granularity) - 10
                if row % 2 == 1:
                    y =  self.granularity * col * 2 + np.random.uniform(0, self.granularity) - 10
                if row % 2 == 0:
                    y =  self.granularity * col * 2 + np.random.uniform(self.granularity, self.granularity * 2) - 10
            # print(x, y)
        else:
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)

        self._goal = np.array([x, y])

    def reset(self):
        self._state = np.array([0, 0], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self._state)

    def reward_function(self, x, y):
        return - (x ** 2 + y ** 2) ** 0.5

    def step(self, action):
        x, y = self._state
        # compute reward, add penalty for large actions instead of clipping them
        x -= self._goal[0]
        y -= self._goal[1]
        # check if task is complete
        done = abs(x) < .01 and abs(y) < .01
        reward = self.reward_function(x, y)
        # move to next state
        self._state = self._state + action
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

    def seed(self, seed):
        np.random.seed = seed
