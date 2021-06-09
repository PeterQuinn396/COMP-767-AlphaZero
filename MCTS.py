class Node():
    def __init__(self, state, model, parent_node):
        self.c = 2  # hyperparameter
        self.state = np.copy(state)
        self.z = None
        self.parent = parent_node
        self.player = 1 if parent_node is None else -parent_node.player  # +1 if first player, -1 if 2nd player

        x = torch.tensor(self.state, device=device, dtype=torch.float32)

        predict = model(x)
        self.initial_probs = predict[:-1].detach().cpu().numpy()  # don't need grad here, this is making training data
        self.value = predict[-1].detach().cpu().numpy()  # nor here

        action_size = predict.size()[0] - 1
        self.Q = np.zeros(action_size)
        self.actions_taken = np.zeros(action_size)
        self.last_action = -1  # used when we pass back through the tree to update Q values


def search(current_node, node_dict, agent, game):  # get the next action for the game

    # an UCB value for each action
    u = current_node.Q + current_node.c * current_node.initial_probs * np.sqrt(np.sum(current_node.actions_taken)) / (
            1 + current_node.actions_taken) + 1e-6  # a small inital prob for each action for numerical stability/simplcity

    mask = game.getLegalActionMask()
    u_masked = mask * u
    u_masked[u_masked == 0] = np.nan  # make the zeros nans so they are ignored by the argmax
    action = np.nanargmax(u_masked)

    current_node.last_action = action
    current_node.actions_taken[action] += 1

    next_state, reward, done = game.step(action)

    if done:  # reached terminal state
        # update q values for the nodes along this path
        node = current_node
        propagate_value_up_tree(node, reward)
        return

    else:  # grow tree and search from the next node
        key = next_state.tobytes()
        if key in node_dict:
            next_node = node_dict[key]
            search(next_node, node_dict, agent, game)
            return
        else:  # this is a new state not currently in the tree, we create the new node and backpropagate its value up the tree
            new_node = make_new_node_in_tree(next_state, node_dict, agent, current_node)
            # we only search the next node if it was already in the tree
            return


def propagate_value_up_tree(leaf_node, value):
    node = leaf_node
    # update Q with moving average
    new_q = node.Q[node.last_action] * (node.actions_taken[node.last_action] - 1) + value * node.player
    new_q /= node.actions_taken[node.last_action]
    node.Q[node.last_action] = new_q

    if leaf_node.parent is None:  # root node
        return
    else:
        node = leaf_node.parent
        propagate_value_up_tree(node, value)


def make_new_node_in_tree(obs, node_dict, agent, last_node):
    new_node = Node(obs, agent, last_node)
    v = new_node.value

    # pass the value of new node up tree and update the Q values
    propagate_value_up_tree(last_node, v)

    # use a dictionary to store the graph
    key = obs.tobytes()
    node_dict[key] = new_node
    return new_node