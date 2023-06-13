import gym_cellular_automata as gymca
from stable_baselines3 import A2C, SAC, PPO, TD3, DQN
import numpy as np
from decimal import Decimal


SQUARE_SHAPE = 20
T_MOVE = 0.025
T_SHOOT = 0.05
T_ANY = 0.025

POS_BULL = None #[SQUARE_SHAPE-1, SQUARE_SHAPE-1]
BULL_POS = None #f'{SQUARE_SHAPE-1}.{SQUARE_SHAPE-1}'

MODEL = PPO
NAME_MODEL = 'PPO'
TOTAL_PROCS = 16
TRAIN_STEPS = 2e6
STEPS_TRESH = int(10e6)
ENV_SEED = 100

NAME_REWARD = 'NumTreesRewardStepOne'
OBS_WRAP = 'ObservationOneHotWrapper_dict_box'

POLICY_TYPE = 'MultiInputPolicy'

params = {
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
    'policy_type': POLICY_TYPE#"MlpPolicy"
}

# prototype mode
ProtoEnv = gymca.prototypes[1]

env = ProtoEnv(nrows=params['nrows'],
               ncols=params['ncols'],
               pos_bull=params['pos_bull'],
               t_move=params['t_move'],
               t_shoot=params['t_shoot'],
               t_any=params['t_any'])

#env = ProtoEnv(nrows=12, ncols=12)
obs = env.reset()
print(obs)

row, col = obs[1][1]

#env.render()

#print(env.grid)
#print(type(env.grid))

anchor_point = [row, col]
print("anchor_point", anchor_point)

# create a sample grid
grid = np.array([[3, 2, 1],
                 [5, 3, 7],
                 [3, 3, 4]])
#print(grid)
# Suppose anchor point is not in any tree
def distance_matrix(grid, anchor_point, t_move, t_shoot, t_any, round_float=3):
    # t_any: time added at any step
    # t_move: time added when the bulldozer moves to its neighborhood (not its same position)
    # t_shoot: time to added to cut a tree.
    t_move = Decimal(t_move)
    t_shoot = Decimal(t_shoot)
    t_any = Decimal(t_any)

    # initialize the distances array with zeros
    distances_array = []
    row, col = anchor_point

    # iterate over all values in the grid that are equal to 3
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 3:
                distances = []
                # calculate the distance from value to all other 3s in the grid
                for k in range(grid.shape[0]):
                    for l in range(grid.shape[1]):
                        if grid[k, l] == 3:
                            d = Decimal(max(abs(i-k), abs(j-l)))
                            d = d * (t_any + t_move)
                            # -------------------------------
                            # We do strong assumptions: As the MFP constrains need d(x,x) = 0,
                            # We need that the anchor point don't be a tree. With this:
                            # * The optimum never digs its own place, then
                            # * we can substitute d(x,x) = t_any + t_shoot by d'(x,x) = 0
                            # -------------------------------
                            if d > 0:
                                d += t_shoot
                            d = round(float(d), round_float)
                            distances.append(d)
                # Calculate the distance to the anchor point
                d = Decimal(max(abs(i - row), abs(j - col)))
                d = d * (t_any + t_move)
                d = round(float(d), round_float)
                # From tree to anchor point we have not to add t_shoot because anchor point is not a tree
                distances.append(d)

                distances_array.append(distances)
    # Add the anchor point
    distances = []
    # calculate the distance from anchor point to all other treed in the grid
    for k in range(grid.shape[0]):
        for l in range(grid.shape[1]):
            if grid[k, l] == 3:
                d = Decimal(max(abs(row - k), abs(col - l)))
                # If the bulldozer moves to the anchor point to a tree, it shoots
                d = d * (t_any + t_move) + t_shoot
                d = round(float(d), round_float)
                distances.append(d)
    # Calculate the distance to the anchor point to itself
    row, col = anchor_point
    d = Decimal(max(abs(row - row), abs(col - col)))
    d = round(float(d), round_float)
    distances.append(d)

    distances_array.append(distances)

    # print the grid and distances
    print("Grid:")
    print(grid)
    print("Distances:")
    print(distances_array)

anchor_point = [1, 2]
distance_matrix(grid, anchor_point, 0,0,1)
distance_matrix(grid,anchor_point,T_MOVE,T_SHOOT,T_ANY)


print("------------- Forest -------------------")
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import lil_matrix

import numpy as np
import random

import numpy as np

import numpy as np

import numpy as np

def create_forest(n):
    # create an empty grid
    forest = np.zeros((n,n), dtype=int)

    # select the root node at the center of the grid
    root = (n//2, n//2)
    forest[root] = 1

    # initialize a queue with the root node
    queue = [root]

    # while there are nodes in the queue
    while queue:
        # get the next node to visit
        curr_node = queue.pop(0)

        # define the neighbor positions
        neighbors = [(curr_node[0]-1, curr_node[1]),
                     (curr_node[0]+1, curr_node[1]),
                     (curr_node[0], curr_node[1]-1),
                     (curr_node[0], curr_node[1]+1)]

        # shuffle the neighbors to ensure a deterministic order
        np.random.shuffle(neighbors)

        # visit the first neighbor that is within the grid and not yet visited
        for neighbor in neighbors:
            if 0 <= neighbor[0] < n and 0 <= neighbor[1] < n and forest[neighbor] == 0:
                forest[neighbor] = 1
                queue.append(neighbor)
                break

    return forest


forest = create_forest(10)
print(forest)


grid_temp = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

arr = np.array(grid_temp)
