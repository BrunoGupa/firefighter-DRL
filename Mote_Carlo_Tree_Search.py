import random
import math

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def add_child(self, child_state):
        child = MCTSNode(child_state, self)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

    def fully_expanded(self):
        return all(child is not None for child in self.children)

    def best_child(self, exploration_param=1.4):
        scores = [
            (child.wins / child.visits) + exploration_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children
        ]
        best_index = max(range(len(scores)), key=lambda x: scores[x])
        return self.children[best_index]

    def rollout(self, rollout_policy):
        current_state = self.state
        while not current_state.is_terminal():
            action = rollout_policy(current_state)
            current_state = current_state.take_action(action)
        return current_state.result()

def monte_carlo_tree_search(root, iters, rollout_policy):
    for _ in range(iters):
        node = root
        state = root.state

        # Select
        while node.fully_expanded() and not state.is_terminal():
            node = node.best_child()
            state = node.state

        # Expand
        if not state.is_terminal():
            action = random.choice(state.available_actions())
            node = node.add_child(state.take_action(action))
            state = node.state

        # Rollout
        result = node.rollout(rollout_policy)

        # Backpropagate
        while node is not None:
            node.update(result)
            node = node.parent

    return root.best_child().state.last_action
