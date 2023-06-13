import gym
import networkx as nx
import numpy as np


# Define the Firefighter problem as a Gym environment
class FirefighterEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(self.graph.nodes)
        self.num_nodes = self.graph.number_of_nodes()
        self.observation_space = gym.spaces.MultiBinary(self.num_nodes)
        self.action_space = gym.spaces.Discrete(self.num_nodes)

    def reset(self):
        self.fire_state = np.zeros(self.num_nodes, dtype=int)
        self.fire_state[0] = 1  # Fire has started at node 1
        return self.fire_state

    def step(self, action):
        next_state = self.fire_state.copy()
        neighbors = list(self.graph.neighbors(self.nodes[action]))
        next_state[action] = 0
        for neighbor in neighbors:
            next_state[self.nodes.index(neighbor)] = 0
        reward = -(next_state.sum() - self.fire_state.sum())
        self.fire_state = next_state
        done = self.fire_state.sum() == 0
        return next_state, reward, done, {}

    def render(self, mode='human'):
        pass


# Define the graph
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5])
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 5)])

# Create the environment
env = FirefighterEnv(G)

# Run a random policy for demonstration purposes
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    state = next_state

# Print the final state
print("Final state:", state)
