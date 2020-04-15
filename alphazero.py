# -*- coding: utf-8 -*-
"""AlphaZero

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Bl4LkvNENrOjkg1eDlS_-g_4E25R8x9O
"""

import torch
import numpy as np

# np.seterr(all='raise')

import time
from torch.nn import Linear
from torch.nn.functional import relu, softmax
from torch import tanh
import copy
from tictactoe import tictactoe
import matplotlib.pyplot as plt

# set up NN

# game state should have player turn as last value

try:
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
except Exception():
    device = torch.device("cpu")  # force cpu

print(f"Using {device}")


class AlphaZero(torch.nn.Module):

    def __init__(self, input_dim, hidden_layer_dim, output_dim):  # should probably use conv layers and residual layers
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = Linear(input_dim, hidden_layer_dim)
        self.fc_layers= torch.nn.ModuleList([Linear(hidden_layer_dim, hidden_layer_dim), Linear(hidden_layer_dim, hidden_layer_dim)])
        self.policy_layer= Linear(hidden_layer_dim, output_dim)
        self.value_layer = Linear(hidden_layer_dim, 1)

    def forward(self, x):
        h = relu(self.input_layer(x))
        for l in self.fc_layers: # all the hidden layers
            h = relu(l(h))
        p = softmax(self.policy_layer(h), dim=-1)  # probs for each action
        _v = self.value_layer(h2)  # predict a value for this state
        v = tanh(_v)

        if len(p.size()) == 2:  # we were doing a batch input of vectors x
            return torch.cat((p, v), dim=1)  # cat along dim 1
        else:  # a single vector x was input, cat along dim 0
            return torch.cat((p, v), dim=0)


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


# simulate many games, build up tree of nodes
def simulate_game(game, agent, search_steps):
    s_list = []
    pi_list = []
    z_list = []

    obs, reward, done = game.reset()

    root = Node(obs, agent, None)
    node_dict = {}
    node_dict[root.state.tobytes()] = root

    current_node = root

    while not done:
        current_state = obs
        key = obs.tobytes()
        if key in node_dict:
            current_node = node_dict[key]
        else:
            current_node = make_new_node_in_tree(current_state, node_dict, agent, current_node)
        current_node.parent = None  # we don't need to backpropagate the values through the tree past this node
        for n in range(search_steps):  # run search_steps trajectories through game
            copy_game = game.copy()
            search(current_node, node_dict, agent,
                   copy_game)  # game will get modified, copy game before calling search or be able to reset state

        # normalize the actions taken during the searching to get probabilities learned during MCTS
        legal_action_mask = game.getLegalActionMask()
        pi = current_node.actions_taken * legal_action_mask

        pi = pi / np.sum(pi)

        s_list.append(current_state)
        pi_list.append(pi)
        z_list.append(0)  # to be filled with game outcome later

        action = np.random.choice(list(range(game.action_space_size)), p=pi)
        # reward should be +1 if 1st player won, -1 if 2nd player won, 0 if tie
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
        s_list, pi_list, z_list = simulate_game(game, agent, search_steps)
        _s = torch.tensor(s_list, device=device).to(dtype=torch.float)
        _pi = torch.tensor(pi_list, device=device).to(dtype=torch.float)
        _z = torch.tensor(z_list, device=device).to(dtype=torch.float)

        s = torch.cat((s, _s))
        pi = torch.cat((pi, _pi))
        z = torch.cat((z, _z))

    return s, pi, z


# set up policy improvement of NN using the simulated games
def improve_model(model, training_data, steps, lr=.001, verbose=False):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    mean_loss = None
    for i in range(steps):
        s = training_data[0]
        pi = training_data[1]
        z = training_data[2]

        y = model(s)
        p = y[:, :-1] + 1e-6  # numerical stability so we don't get log(0)
        v = y[:, -1]

        loss = (v - z) * (v - z) - torch.sum(pi * torch.log(p), dim=-1)

        value_loss = torch.mean((v - z) * (v - z))
        policy_loss = -torch.mean(torch.sum(pi * torch.log(p), dim=-1))
        mean_loss = value_loss + policy_loss

        if torch.isnan(mean_loss):
            print("getting nans")

        if verbose:
            print(f"Step {i}, Loss: {mean_loss}")

        optimizer.zero_grad()
        mean_loss.backward()
        optimizer.step()


    return mean_loss.detach().numpy(), value_loss.detach().numpy(), policy_loss.detach().numpy()



# set up playing games between old agent and updated agent, keeping the winner
def play_game(p1, p2, game, greedy=False):
    obs, reward, done = game.reset()
    model_playing = p1
    while not done:
        x = torch.tensor(obs, device=device, dtype=torch.float32)
        y = model_playing(x)
        v = y[-1]
        p = y[:-1]

        p = p.detach().cpu().numpy()
        mask = game.getLegalActionMask()
        p = p * mask

        p = p / np.sum(p)  # normalize probs

        if greedy:  # this option is bad since the same game gets played everytime
            action = np.argmax(p)
        else:
            action = np.random.choice(list(range(game.action_space_size)), p=p)

        obs, reward, done = game.step(action)

        if done:
            return reward  # +1 if p1 wins, -1 if p2 wins, 0 if draw

        # alternate turns
        if model_playing is p1:
            model_playing = p2
        else:
            model_playing = p2


def compare_agents(old_agent, new_agent, game, num_games, greedy=False):  # num_games should be even

    new_wins = 0
    old_wins = 0
    ties = 0

    for i in range(num_games // 2):
        r = play_game(old_agent, new_agent, game, greedy=greedy)
        if r == 1:
            old_wins += 1
        elif r == -1:
            new_wins += 1
        elif r == 0:
            ties += 1
        else:
            raise RuntimeError("Got bad return from game")

    for i in range(num_games // 2):
        r = play_game(new_agent, old_agent, game, greedy=greedy)
        if r == 1:
            new_wins += 1
        elif r == -1:
            old_wins += 1
        elif r == 0:
            ties += 1
        else:
            raise RuntimeError("Got bad return from game")
    return new_wins, old_wins, ties


# training loop
def main():
    game = tictactoe()
    input_size = game.obs_space_size
    output_size = game.action_space_size
    hidden_layer_size = 256
    agent = AlphaZero(input_size, hidden_layer_size, output_size)

    agent.to(device)

    iterations = 800  # how many times we want to make training data, update a model
    num_games = 25  # play certain number of games to generate examples each iteration
    search_steps = 25  # for each step of each game, consider this many possible outcomes
    optimization_steps = 100  # once we have generated training data, how many epochs do we do on this data

    num_faceoff_games = 40  # when comparing updated model and old model, how many games they play to determine winner

    best_loss = .15
    training_data = None
    new_agent = None
    for itr in range(iterations):
        print(f"Starting iteration {itr + 1} / {iterations}")

        print(f"Generating training data...")
        if training_data is None:
            training_data = generate_training_data(game, num_games, search_steps, agent)
        else:  # we generate more data if our improved agent fails to beat the old one
            print("Generating additional data...")
            more_data = generate_training_data(game, num_games, search_steps, agent)
            s = torch.cat((training_data[0], more_data[0]))
            pi = torch.cat((training_data[1], more_data[1]))
            z = torch.cat((training_data[2], more_data[2]))
            training_data = [s, pi, z]

        print(f"Generated training data: {training_data[0].size(0)} states")

        if new_agent is None:  # keep the progress on the new model if we failed to beat the old one
            new_agent = copy.deepcopy(agent)

        print(f"Improving model...")
        loss, value_loss, policy_loss = improve_model(new_agent, training_data, optimization_steps, lr=.001, verbose=False)
        print(f"Finished improving model, total loss: {loss}, value loss: {value_loss}, policy loss: {policy_loss}")

        if loss < best_loss:  # model is pretty good. lets stop and check it out
            torch.save(agent.state_dict(), f"saved_models/tictactoe_agent_{loss}.pt")
            best_loss = loss

        print(f"Comparing agents...")
        new_wins, old_wins, ties = compare_agents(agent, new_agent, game, num_faceoff_games)
        print(f"New wins: {new_wins}, old wins: {old_wins}, ties: {ties}")
        if new_wins > old_wins:
            agent = new_agent
            new_agent = None
            training_data = None

        elif training_data[0].size(0) > 3000:  # prevent getting stuck with a bad model, reset to old model and new data
            new_agent = None
            training_data = None
        training_data = None  # always reset training data, helps?

    print("Saving agent")
    torch.save(agent.state_dict(), "saved_models/tictactoe_agent.pt")
    return agent


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    elapsed = end - start
    h = elapsed // 3600
    m = (elapsed - h * 3600) // 60
    s = elapsed % 60
    print(f"Time: {h} h {m} m {s} s")
