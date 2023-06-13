import random

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

    def fully_expanded(self):
        return all(child is not None for child in self.children)

    def best_child(self, c=1.4):
        scores = [
            (child.wins / child.visits) + c * (2 * log(self.visits) / child.visits)**0.5
            for child in self.children
        ]
        max_index = scores.index(max(scores))
        return self.children[max_index]

    def rollout(self, rollout_policy):
        current_rollout_state = self.state
        while not current_rollout_state.is_terminal():
            action = rollout_policy(current_rollout_state)
            current_rollout_state = current_rollout_state.take_action(action)
        return current_rollout_state.result()


def monte_carlo_tree_search(root, iters, rollout_policy):
    for i in range(iters):
        node = root
        state = root.state

        # Select
        while node.fully_expanded() and not state.is_terminal():
            node = node.best_child()
            state = node.state

        # Expand
        if not state.is_terminal():
            action = random.choice(state.available_actions())
            child = node.add_child(state.take_action(action))
            node = child
            state = node.state

        # Rollout
        result = node.rollout(rollout_policy)

        # Backpropagate
        while node is not None:
            node.update(result)
            node = node.parent

    return root.best_child().state.last_action
