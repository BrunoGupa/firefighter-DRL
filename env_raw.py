import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define the graph
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5])
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 5)])

# Define the fire state
fire_state = np.zeros(G.number_of_nodes())
fire_state[0] = 1  # Fire has started at node 1

# Define the action space
actions = [i for i in range(G.number_of_nodes())]


# Define the transition function
def transition_function(state, action):
    next_state = state.copy()
    neighbors = list(G.neighbors(action))
    next_state[action] = 0
    for neighbor in neighbors:
        next_state[neighbor] = 0
    return next_state


# Define the reward function
def reward_function(state, next_state):
    return -(next_state.sum() - state.sum())


# Define the episode
def episode(policy, state):
    episode_states = [state]
    episode_actions = []
    episode_rewards = []

    while state.sum() > 0:
        action = policy(state)
        next_state = transition_function(state, action)
        reward = reward_function(state, next_state)

        episode_states.append(next_state)
        episode_actions.append(action)
        episode_rewards.append(reward)

        state = next_state

    return episode_states, episode_actions, episode_rewards


# Define a random policy for demonstration purposes
def random_policy(state):
    return np.random.choice(actions)


# Run the episode
states, actions, rewards = episode(random_policy, fire_state)

# Plot the graph
nx.draw(G, with_labels=True)
plt.show()

# Print the results
print("States:", states)
print("Actions:", actions)
print("Rewards:", rewards)
