import gym
import gym_cellular_automata as gymca

import numpy as np

from stable_baselines3 import A2C, SAC, PPO, TD3, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.utils import set_random_seed


import gym
import numpy as np

import gym
import numpy as np

class NumTreesRewardStepOne(gym.RewardWrapper):
    """
    A reward wrapper that returns the negative number of burnt trees in a grid.

    Assumes that the tree cells are labeled as 3 in the grid.

    The reward is calculated as the negative change in the count of non-burnt trees
    from the previous step to the current step.

    Args:
        env: A gym environment with a 2D grid of cells representing trees.
    """

    def __init__(self, env):
        super().__init__(env)
        self.num_trees = np.sum(np.where(self.grid == 3, 1, 0))

    def reward(self, rew):
        """
        Calculates the negative change in the count of non-burnt trees.

        Returns:
            A negative reward equal to the change in the count of non-burnt trees.
        """
        new_num_trees = np.sum(np.where(self.grid == 3, 1, 0))
        delta = self.num_trees - new_num_trees
        self.num_trees = new_num_trees
        return -delta


class NumTreesRewardStep(gym.RewardWrapper):
    #Returns the number of fires  at the end of each cycle
    def __init__(self, env):
        super().__init__(env)
        self.actual_time = 0

    def reward(self, rew):

        time = self.context[-1]
        if time - self.actual_time < 0:
            actual_burnt = np.where(self.grid == 25, 1, 0)
            unique, counts = np.unique(actual_burnt, return_counts=True)

            self.actual_time = time
            try:
                return - counts[1] / 40 #the greater loss? #(counts[0] + counts[1])
            except IndexError:
                # There are not any fire. Means Done
                self.actual_time = 0
                return 0
        self.actual_time = time

        return 0

class NumTreesRewardStep(gym.RewardWrapper):
    #Returns the number of fires  at the end of each cycle
    def __init__(self, env):
        super().__init__(env)
        self.actual_time = 0

    def reward(self, rew):

        time = self.context[-1]
        if time - self.actual_time < 0:
            actual_burnt = np.where(self.grid == 25, 1, 0)
            unique, counts = np.unique(actual_burnt, return_counts=True)

            self.actual_time = time
            try:
                return - counts[1] / 40 #the greater loss? #(counts[0] + counts[1])
            except IndexError:
                # There are not any fire. Means Done
                self.actual_time = 0
                return 0
        self.actual_time = time

        return 0


class NumTreesReward(gym.RewardWrapper):
    #Returns the number of trees at the end of the episode
    def __init__(self, env, normalized=False):
        super().__init__(env)
        self.normalized = normalized

    def reward(self, rew):
        if self.done:
            #print(self.grid)
            grid_tree = np.where(self.grid == 3, 1, 0)
            unique, counts = np.unique(grid_tree, return_counts=True)

            try:
                if self.normalized:
                    return counts[1] / 38 #(counts[0] + counts[1])
                else:
                    return counts[1]
            except IndexError:
                # There are not any tree
                return 0
        return 0
