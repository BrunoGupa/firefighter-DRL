import gym
import numpy as np

class ObservationOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.MultiBinary([3, shape, shape]),
            "time": gym.spaces.Box(low=0, high=1, shape=(1,))
        })

    def observation(self, obs):
        #print('obs inside')
        #print(obs)
        grid = obs[0]
        bulldozer_pos = obs[1][1]

        grid_pos = np.zeros((self.shape, self.shape))
        grid_pos[bulldozer_pos[0], bulldozer_pos[1]] = 1


        grid_tree = np.where(grid == 3, 1, 0)
        grid_fire = np.where(grid == 25, 1, 0)

        # Time_Space
        time = obs[1][-1]
        time = 1 if time >= 1 else time

        return {
            "grid": np.stack([grid_tree, grid_fire, grid_pos]),
            "time": np.array([time])
        }





# import gym
# import numpy as np
#
#
# class ObservationOneHotWrapper(gym.ObservationWrapper):
#     def __init__(self, env, shape):
#         super().__init__(env)
#         self.shape = shape
#         #self.observation_space = gym.spaces.MultiBinary([3, shape, shape])
#
#         self.observation_space = gym.spaces.Dict(
#             {"grid": gym.spaces.MultiBinary([3, shape, shape]),
#              "time": gym.spaces.Box(low=0, high=1, shape=(1, 1))
#             }
#         )
#
#     def observation(self, obs):
#         print('obs inside')
#         print(obs)
#         grid = obs[0]
#         row, col = obs[1][1]
#         #print('pos:', row, col)
#         grid_pos = np.zeros((self.shape, self.shape))
#         grid_pos[row, col] = 1
#         grid_tree = np.where(grid == 3, 1, 0)
#         grid_fire = np.where(grid == 25, 1, 0)
#
#         #Time_Space
#         time = obs[1][-1]
#         time = 1 if time >= 1 else time
#
#         return {"grid": [grid_tree, grid_fire, grid_pos],
#                 "time": time}
