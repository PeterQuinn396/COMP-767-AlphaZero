# -*- coding: utf-8 -*-
"""AlphaZero

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Bl4LkvNENrOjkg1eDlS_-g_4E25R8x9O
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.nn import Linear, ReLU, Softmax, Sigmoid
from torch.nn.functional import relu, softmax
from torch import sigmoid
import copy
from tictactoe import tictactoe

# set up NN

# game state should have player turn as last value
device = "cuda:0" if torch.cuda.is_available else "cpu"
device = "cpu"
print(f"Using {device}")


class AlphaZero(torch.nn.Module):

    def __init__(self, input_dim, hidden_layer_dim, output_dim):  # should probably use conv layers and residual layers
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = Linear(input_dim, hidden_layer_dim)
        self.fc2 = Linear(hidden_layer_dim, hidden_layer_dim)
        self.fc3 = Linear(hidden_layer_dim, output_dim)
        self.value_layer = Linear(hidden_layer_dim, 1)

    def forward(self, x):
        h1 = relu(self.fc1(x))
        h2 = relu(self.fc2(h1))
        p = softmax(self.fc3(h2))  # probs for each action
        _v = self.value_layer(h2)  # predict a value for this state
        v = 2 * sigmoid(_v) - 1  # maps between [-1,1]

        if len(p.size()) == 2:  # we were doing a batch input of vectors x
            return torch.cat((p, v), dim=1)  # cat along dim 1
        else:  # a single vector x was input, cat along dim 0
            return torch.cat((p, v), dim=0)


# set up Monte Carlo Tree Searching (MCTS)
states_tensor = None  # [num_of_states_sampled, dim_of_state]
estimated_probs_tensor = None  # [num_of_states_sampled, probs_of_taking_each_action]
outcomes_tensor = None  # [num_states_sampled, final_outcome_of_game]


class Node():
    def __init__(self, state, model, parent_node):
        self.c = .1  # hyperparameter
        self.state = np.copy(state)
        self.z = None
        self.parent = parent_node
        self.player = 1 if parent_node is None else -parent_node.player  # +1 if first player, -1 if 2nd player

        x = torch.tensor(self.state, device=device, dtype=torch.float32)
        predict = model(x)
        self.initial_probs = predict[:-1].detach().numpy()  # don't need grad here, this is making training data
        self.value = predict[-1].detach().detach().numpy()  # nor here

        action_size = predict.size()[0] - 1
        self.Q = np.zeros(action_size)
        self.actions_taken = np.zeros(action_size)
        self.last_action = -1  # used when we pass back through the tree to update Q values


def search(current_node, node_list, agent, game):  # get the next action for the game

    # an UCB value for each action
    u = current_node.Q + current_node.c * current_node.initial_probs * np.sqrt(np.sum(current_node.actions_taken)) / (
            1 + current_node.actions_taken)
    u_ordered = np.flip(np.argsort(u))  # order the actions in term of their max expected return

    num_actions = u.shape[0]
    i = 0
    action = -1
    while i < num_actions:  # determine the optimal legal action

        action = u_ordered[i]
        if game.isLegalAction(action):
            # print(f"found legal action: {action}")
            break
        else:
            i += 1

    if i >= num_actions:
        print("no legal actions found --- Big Problem!!! ----")
        print(f"action expected values: {u}")
        game.render()
        exit(-1)

    current_node.last_action = action
    current_node.actions_taken[action] += 1

    next_state, reward, done = game.step(action)

    if done:  # reached terminal state
        # update q values for the nodes along this path
        node = current_node
        while not (node.parent is None):
            # update Q with moving average
            new_q = node.Q[node.last_action] * (node.actions_taken[node.last_action] - 1) + reward * node.player
            new_q /= node.actions_taken[node.last_action]
            node.Q[node.last_action] = new_q
            # move up the tree
            node = node.parent
        return

    else:  # grow tree and search from the next node

        for node in node_list:
            if np.array_equal(node.state, next_state):
                # we have the next state in the tree already
                next_node = node
                break
        else:  # this is a new state not currently in the tree
            new_node = make_new_node_in_tree(next_state, node_list, agent, current_node)
            next_node = new_node

        search(next_node, node_list, agent, game)


def make_new_node_in_tree(obs, node_list, agent, last_node):
    new_node = Node(obs, agent, last_node)
    # pass the value of new node up tree and update the Q values
    node = new_node
    v = new_node.value
    while not (node.parent is None):
        # update Q with moving average based on the bootstrapped value function for the new node
        new_q = node.Q[node.last_action] * (node.actions_taken[node.last_action] - 1) + v * node.player
        new_q /= node.actions_taken[node.last_action]
        node.Q[node.last_action] = new_q
        # move up the tree
        node = node.parent

    node_list.append(new_node)
    return new_node


# simulate many games, build up tree of nodes
def simulate_game(game, agent, search_steps):
    # example_list = []  # a tuple of [state,action_probs,outcome], will be considered our target values when training
    s_list = []
    pi_list = []
    z_list = []

    obs, reward, done = game.reset()

    root = Node(obs, agent, None)
    node_list = []
    node_list.append(root)

    current_node = root

    while not done:
        current_state = obs

        for node in node_list:
            if np.array_equal(node.state, current_state):
                current_node = node
                break
        else:
            current_node = make_new_node_in_tree(current_node, node_list, agent, current_node)

        for n in range(search_steps):  # run search_steps trajectories through game
            copy_game = game.copy()
            search(current_node, node_list, agent,
                   copy_game)  # game will get modified, copy game before calling search or be able to reset state

        # normalize the actions taken during the searching to get probabilities learned during MCTS
        legal_action_mask = game.getLegalActionMask()
        pi = current_node.actions_taken * legal_action_mask
        pi = pi / np.sum(pi)

        s_list.append(current_state)
        pi_list.append(pi)
        z_list.append(0)  # to be filled with game outcome later

        action = np.random.choice(list(range(game.action_space_size)), p=pi)

        # reward should be +1 if 1st player won, -1 if 2nd player won
        obs, reward, done = game.step(action)

        if done:
            for i, z in enumerate(z_list):  # fill in the outcome for every tuple along this path
                z_list[i] = reward

    return s_list, pi_list, z_list


def generate_training_data(game, num_games, search_steps, agent):
    # intialize tensors after simulating first game
    s_list, pi_list, z_list = simulate_game(game, agent, search_steps)

    s = torch.tensor(s_list, device=device).to(dtype=torch.float)
    pi = torch.tensor(pi_list, device=device).to(dtype=torch.float)
    z = torch.tensor(z_list, device=device).to(dtype=torch.float)

    for _ in range(num_games - 1):  # make example list into tensors for training
        ex = simulate_game(game, agent, search_steps)
        _s = torch.tensor(ex[0], device=device).to(dtype=torch.float)
        _pi = torch.tensor(ex[1], device=device).to(dtype=torch.float)
        _z = torch.tensor(ex[2], device=device).to(dtype=torch.float)

        s = torch.cat((s, _s))
        pi = torch.cat((pi, _pi))
        z = torch.cat((z, _z))

    return s, pi, z


# set up policy improvement of NN using the simulated games
def improve_model(model, training_data, steps, lr=.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(steps):
        s = training_data[0]
        pi = training_data[1]
        z = training_data[2]

        y = model(s)
        p = y[:, :-1]
        v = y[:, -1]

        loss = (v - z) * (v - z) - torch.sum(pi * torch.log(p), dim=-1)
        loss = torch.sum(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# set up playing games between old agent and updated agent, keeping the winner
def play_game(p1, p2, game):
    obs, reward, done = game.reset()
    model_playing = p1
    while not done:
        x=torch.tensor(obs, device=device, dtype=torch.float32)
        y = model_playing(x)
        v = y[-1]
        p = y[:-1]

        p = p.detach().numpy()
        mask = game.getLegalActionMask()
        p = p * mask

        p = p/np.sum(p) # normalize probs

        action = np.random.choice(list(range(game.action_space_size)), p=p)
        obs, reward, done = game.step(action)

        if done:
            return reward  # +1 if p1 wins, -1 if p2 wins, 0 if draw

        # alternate turns
        if model_playing is p1:
            model_playing = p2
        else:
            model_playing = p2


def compare_agents(old_agent, new_agent, game, num_games):  # num_games should be even

    new_wins = 0
    old_wins = 0
    total_games = 0
    for i in range(num_games // 2):
        r = play_game(old_agent, new_agent, game)
        if r == 1:
            old_wins += 1
        elif r == -1:
            new_wins += 1

    for i in range(num_games // 2):
        r = play_game(new_agent, old_agent, game)
        if r == 1:
            new_wins += 1
        elif r == -1:
            old_wins += 1

    return new_wins, old_wins




# training loop
def main():
    game = tictactoe()
    input_size = game.obs_space_size
    output_size = game.action_space_size
    hidden_layer_size = 128
    agent = AlphaZero(input_size, hidden_layer_size, output_size)

    iterations = 100  # how many times we want to make training data, update a model
    num_games = 50  # play certain number of games to generate examples each iteration
    search_steps = 1  # for each step of each game, consider 4 possible outcomes
    optimization_steps = 100  # once we have generated training data, how many epochs do we do on this data
    num_faceoff_games = 20  # when comparing updated model and old model, how many games they play to determine winner

    training_data = None
    new_agent = None
    for itr in range(iterations):
        print(f"Starting iteration {itr + 1} / {iterations}")

        print(f"Generating training data...")
        if training_data is None:
            training_data = generate_training_data(game, num_games, search_steps, agent)
        else:  # we generate more data if our improved agent fails to beat the old one
            more_data = generate_training_data(game, num_games, search_steps, agent)
            training_data = torch.cat((training_data, more_data))
        print(f"Generated training data: {training_data[0].size(0)} states")

        if new_agent is None:  # keep the progress on the new model if we failed to beat the old one
            new_agent = copy.deepcopy(agent)

        print(f"Improving model...")
        improve_model(new_agent, training_data, optimization_steps)
        print(f"Finished improving model.")

        print(f"Comparing agents...")
        new_wins, old_wins = compare_agents(agent, new_agent, game, num_faceoff_games)
        print(f"New wins: {new_wins}, old wins: {old_wins}")
        if new_wins > old_wins:
            agent = new_agent
            new_agent = None
            training_data = None


if __name__ == "__main__":
    main()