import gym
import gym_cellular_automata as gymca

import numpy as np
import os


from stable_baselines3 import A2C, SAC, PPO, TD3, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env


SQUARE_SHAPE = 10
T_MOVE = 0.01
T_SHOOT = 0.25
BULL_POS = 8.8

# we stablish t_any = T_move

MODEL = PPO

class ObservationOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(shape=(3, SQUARE_SHAPE, SQUARE_SHAPE), low=0, high=1)

    def observation(self, obs):
        #print('obs inside')
        #print(obs)
        grid = obs[0]
        row, col = obs[1][1]
        #print('pos:', row, col)
        grid_pos = np.zeros((SQUARE_SHAPE,SQUARE_SHAPE))
        grid_pos[row, col] = 1
        grid_tree = np.where(grid == 3, 1, 0)
        grid_fire = np.where(grid == 25, 1, 0)
        return grid_tree, grid_fire, grid_pos


class NumTreesReward(gym.RewardWrapper):
    #Returns the number of trees at the end of the episode
    def __init__(self, env):
        super().__init__(env)

    def reward(self, rew):
        if self.done:
            #print(self.grid)
            grid_tree = np.where(self.grid == 3, 1, 0)
            unique, counts = np.unique(grid_tree, return_counts=True)
            try:
                return int(counts[1])
            except IndexError:
                # There are not any tree
                return 0
        return 0


# Create save dir
save_dir = "test_models"
os.makedirs(save_dir, exist_ok=True)


#ProtoEnv = gymca.prototypes[1]
#env = ProtoEnv(nrows=SQUARE_SHAPE, ncols=SQUARE_SHAPE)


env_id = gymca.envs[1]
#env = gym.make(env_id)
#env = make_atari_env(env_id, n_envs=16)
#env = VecFrameStack(env, n_stack=4)


env = ObservationOneHotWrapper(env)
env = NumTreesReward(env)



env.reset()
env.render()
# Frame-stacking with 4 frames

env = Monitor(env)
env = DummyVecEnv([lambda: env])
#env = make_vec_env(env, n_envs=4)

model = MODEL("MlpPolicy", env, verbose=1).learn(int(1000))
#model = PPO('MlpPolicy', 'Pendulum-v1', verbose=0).learn(8000)


#The model will be saved under the shape of the square
model.save(save_dir + f"/{MODEL}_{SQUARE_SHAPE}-{T_MOVE}-{T_SHOOT}-{BULL_POS}")

