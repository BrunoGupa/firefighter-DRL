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

from wrappers import ObservationOneHotWrapper
from reward_wrappers import NumTreesRewardEpisodic
import os


import imageio

# Play
ESSAYS = 1
train_steps = '4m'

MODEL_NAME = 'clean-microwave-19'

SQUARE_SHAPE = 10
T_MOVE = 0.025
T_SHOOT = 0.05
POS_BULL = None #[SQUARE_SHAPE-1, SQUARE_SHAPE-1]
BULL_POS = None #f'{SQUARE_SHAPE-1}.{SQUARE_SHAPE-1}'
T_ANY = 0.025

MODEL = PPO
NAME_MODEL = 'PPO'
TOTAL_PROCS = 16
TRAIN_STEPS = 4e6
STEPS_TRESH = int(10e6)
ENV_SEED = 100

NAME_REWARD = 'NumTreesRewardEpisodic_normalized100'
OBS_WRAP = 'ObservationOneHotWrapper_dict_box'

POLICY_TYPE = 'MultiInputPolicy'

dir_name = f"{NAME_MODEL}_{SQUARE_SHAPE}-{T_MOVE}-{T_SHOOT}-{BULL_POS}"





ProtoEnv = gymca.prototypes[1]
env = ProtoEnv(nrows=SQUARE_SHAPE,
               ncols=SQUARE_SHAPE,
               pos_bull=POS_BULL,
               t_move=T_MOVE,
               t_shoot=T_SHOOT,
               t_any=T_MOVE)

env = ObservationOneHotWrapper(env, shape=SQUARE_SHAPE)
env = NumTreesRewardEpisodic(env)

# Load Model
save_dir = "modelos_chingones/"
model = MODEL.load(save_dir + MODEL_NAME, env=env)

print('Im hereee')
obs = env.reset()
print('obs')

print(obs)
#action = model.predict(obs, deterministic=True)
#print(action)
for essay in range(ESSAYS):
    total_reward = 0.0
    done = False
    step = 0
    threshold = 15
    env.reset()
    #env.render()

    images = []
    obs = env.reset()
    img = env.render(rgb=True)

    while not done: # and step < threshold:
        print(step)
        images.append(img)
        action = model.predict(obs, deterministic=True)[0]
        #action = env.action_space.sample()
        print('action', action)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        img = env.render(rgb=True)

    print('Done', done, step)
    print('Total_reward', total_reward)

    save_dir_gif = f"gifs"
    os.makedirs(save_dir_gif, exist_ok=True)

    gif_name = save_dir_gif + f"/{dir_name}" + f'-{train_steps}' + f"-{str(essay)}"
    imageio.mimsave(f"{gif_name}.gif", images, fps=3)
