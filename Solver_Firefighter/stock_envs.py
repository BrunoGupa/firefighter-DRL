from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from wrappers import ObservationOneHotWrapper
from reward_wrappers import NumTreesReward, NumTreesRewardStepOne


def make_env(env_id, rank, params: dict, monitor=True):
    """
    Utility function for multiprocessed env.

    :param params:
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # Important: use a different seed for each environment

        env = env_id(nrows=params['nrows'],
               ncols=params['ncols'],
               pos_bull=params['pos_bull'],
               pos_fire=params['pos_fire'],
               t_move=params['t_move'],
               t_shoot=params['t_shoot'],
               t_any=params['t_any'])
        env = ObservationOneHotWrapper(env, params['nrows'])
        env.reset()
        env = NumTreesReward(env, normalized=True)
        #env = NumTreesRewardStepOne(env)
        if monitor:
            env = Monitor(env)

        env.seed(params['env_seed'] + rank)

        return env
    set_random_seed(params['env_seed'])
    return _init
