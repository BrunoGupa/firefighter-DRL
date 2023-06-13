import gym_cellular_automata as gymca
from stable_baselines3 import A2C, SAC, PPO, TD3, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import os

import wandb
from wandb.integration.sb3 import WandbCallback


from stock_envs import make_env


SQUARE_SHAPE = 10
T_MOVE = 0.025
T_SHOOT = 0.05
POS_BULL = None #[SQUARE_SHAPE-1, SQUARE_SHAPE-1]
BULL_POS = None #f'{SQUARE_SHAPE-1}.{SQUARE_SHAPE-1}'
T_ANY = 0.025

MODEL = PPO
NAME_MODEL = 'PPO'
TOTAL_PROCS = 16
TRAIN_STEPS = 1000
STEPS_TRESH = int(10e6)
ENV_SEED = 100

NAME_REWARD = 'NumTreesReward_normalized'
OBS_WRAP = 'ObservationOneHotWrapper'

parameters = {
    'nrows': SQUARE_SHAPE,
    'ncols': SQUARE_SHAPE,
    'pos_bull': POS_BULL,
    't_move': T_MOVE,
    't_shoot': T_SHOOT,
    't_any': T_ANY,
    #'model': MODEL,
    'name_model': NAME_MODEL,
    'total_proces': TOTAL_PROCS,
    'steps_threshold': STEPS_TRESH,
    'env_seed': ENV_SEED,
    'name_reward': NAME_REWARD,
    'obs_wrapper': OBS_WRAP,
    'policy_type': "MlpPolicy"
}


run = wandb.init(
    project="bulldozer",
    config=parameters,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    #save_code=True,  # optional
)


ProtoEnv = gymca.prototypes[1]
env = DummyVecEnv([make_env(ProtoEnv, i + TOTAL_PROCS, parameters) for i in range(TOTAL_PROCS)])
model = MODEL(parameters['policy_type'],
              env,
              verbose=1,
              tensorboard_log=f'runs/{run.id}')

model.learn(total_timesteps=int(TRAIN_STEPS),
            callback=WandbCallback(model_save_path=f'models/{run.id}'),
            )

run.finish()

# Create save dir
# Automatic saved if Wandb is used.
#dir_name = f"{NAME_MODEL}_{SQUARE_SHAPE}-{T_MOVE}-{T_SHOOT}-{BULL_POS}"


#save_dir = f"models_{SQUARE_SHAPE}"
#os.makedirs(save_dir, exist_ok=True)
#model.save(save_dir + f"/{dir_name}")
env.close()
