import gym_cellular_automata as gymca
from stable_baselines3 import A2C, SAC, PPO, TD3, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import os

import wandb
from wandb.integration.sb3 import WandbCallback

from stock_envs import make_env
import os

move_values = {
    0: {"T_MOVE": 0.025, "T_SHOOT": 0.1, "T_ANY": 0.025},
    1: {"T_MOVE": 0.05, "T_SHOOT": 0.3, "T_ANY": 0.05},
    2: {"T_MOVE": 0.1, "T_SHOOT": 0.5, "T_ANY": 0.05},
    3: {"T_MOVE": 0.5, "T_SHOOT": 0.5, "T_ANY": 0.05},
    4: {"T_MOVE": 0.1, "T_SHOOT": 0.15, "T_ANY": 0.03, "D": [4]},
    5: {"T_MOVE": 0.2, "T_SHOOT": 0.025, "T_ANY": 0.025, "D": [5]},
    6: {"T_MOVE": 0.3, "T_SHOOT": 0.1, "T_ANY": 0.025, "D": [3]},
    7: {"T_MOVE": 0.4, "T_SHOOT": 0.05, "T_ANY": 0.05, "D": [3]},
}
for POS_FIRE in [(5, 0), (4, 5)]:
    for i in range(8):
        if i == 0 and POS_FIRE == (5, 0):
            continue

        save_dir = "graph_models"
        run_name = "avatar-fire-master-{}-{}".format(i, "a" if POS_FIRE == (5, 0) else "b")
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

        print("---------Running model: {}--------------------".format(run_name))
        SQUARE_SHAPE = 10
        T_MOVE = move_values[i]["T_MOVE"]
        T_SHOOT = move_values[i]["T_SHOOT"]
        T_ANY = move_values[i]["T_ANY"]
        POS_BULL = [SQUARE_SHAPE-1, SQUARE_SHAPE-1]
        BULL_POS = f'{SQUARE_SHAPE-1}.{SQUARE_SHAPE-1}'
        #POS_FIRE = (5, 0)


        MODEL = PPO
        NAME_MODEL = 'PPO'
        TOTAL_PROCS = 8
        TRAIN_STEPS = 3e6 #+ i * 4e5
        STEPS_TRESH = int(3e6)
        ENV_SEED = 100

        NAME_REWARD = 'NumTreesRewardStepOne'
        OBS_WRAP = 'ObservationOneHotWrapper_dict_box'

        POLICY_TYPE = 'MultiInputPolicy'

        parameters = {
            'nrows': SQUARE_SHAPE,
            'ncols': SQUARE_SHAPE,
            'pos_bull': POS_BULL,
            'pos_fire': POS_FIRE,
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
            'policy_type': POLICY_TYPE#"MlpPolicy"
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

        model.save(os.path.join(save_dir, run_name))


        run.finish()

        # Create save dir
        # Automatic saved if Wandb is used.
        #dir_name = f"{NAME_MODEL}_{SQUARE_SHAPE}-{T_MOVE}-{T_SHOOT}-{BULL_POS}"


        #save_dir = f"models_{SQUARE_SHAPE}"
        #os.makedirs(save_dir, exist_ok=True)
        #model.save(save_dir + f"/{dir_name}")
        env.close()
