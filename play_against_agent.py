from alphazero import AlphaZero, AlphaZeroResidual, AlphaZeroConv, play_against_heuristics
from tictactoe import tictactoe
import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"Using {device}")


def load_and_play(filename, agent_plays=1, use_heuristic_agent=False):
    game = tictactoe()
    input_size = game.obs_space_size
    output_size = game.action_space_size
    hidden_layer_size = 128

    if use_heuristic_agent:
        agent = None
    else:
        agent = AlphaZeroResidual(input_size, hidden_layer_size, output_size)
        # agent = AlphaZeroResidual(input_size, hidden_layer_size, output_size)
        # agent = AlphaZeroConv(input_size, hidden_layer_size, output_size)
        agent.load_state_dict(torch.load(filename, map_location=device))
    agent.eval()
    play_with_agent(agent, verbose=True, agent_plays=agent_plays, use_heuristic_agent=use_heuristic_agent)


def get_agent_action(agent, game, state, verbose=False):
    mask = game.getLegalActionMask()
    x = torch.tensor(state, device=device, dtype=torch.float)
    y = agent(x)
    pi = y[:-1]
    v = y[-1].detach().numpy()
    _pi = pi.detach().numpy()
    _pi = _pi * mask
    _pi = _pi / np.sum(_pi)
    action = np.argmax(_pi)

    if verbose:
        print(f"Value: {v}, Probs: {_pi}, action: {action}")

    return action


def play_with_agent(agent, verbose=False, agent_plays=1, use_heuristic_agent=False):
    game = tictactoe()
    obs, reward, done = game.reset()
    game.render()
    while not done:
        if agent_plays == 1:

            if use_heuristic_agent:
                agent_action = game.get_computer_move()
            else:
                agent_action = get_agent_action(agent, game, obs, verbose=verbose)

            obs, reward, done = game.step(agent_action)
            game.render()
            # player takes turn
            if not done:

                action = int(input("Input play space (0-8): "))

                while not game.isLegalAction(action):
                    print("Illegal action")
                    action = int(input("Input play space (0-8): "))

                obs, reward, done = game.step(action)
                game.render()

        elif agent_plays == 2:

            action = int(input("Input play space (0-8): "))

            while not game.isLegalAction(action):
                print("Illegal action")
                action = int(input("Input play space (0-8): "))

            obs, reward, done = game.step(action)
            game.render()
            if not done:
                if use_heuristic_agent:
                    agent_action = game.get_computer_move()
                else:
                    agent_action = get_agent_action(agent, game, obs, verbose=verbose)
                obs, reward, done = game.step(agent_action)
                game.render()

        else:
            raise Exception("Invalid value for agent plays parameter. Choose 1 or 2")

    if reward == 0:
        print("Tie game")
    elif (reward == 1 and agent_plays == 1) or (reward == -1 and agent_plays == 2):
        print("Agent won")
    else:
        print("You won!")



def agent_play_against_heuristics(filename):
    game = tictactoe()
    game.reset()
    input_size = game.obs_space_size
    output_size = game.action_space_size
    hidden_layer_size = 128
    # agent = AlphaZero(input_size, hidden_layer_size, output_size)
    # agent = AlphaZeroResidual(input_size, hidden_layer_size, output_size)

    agent = AlphaZeroConv(input_size, hidden_layer_size, output_size)
    agent.load_state_dict(torch.load(filename, map_location=device))


if __name__ == "__main__":
    load_and_play("saved_models/tictactoe_agent.pt", agent_plays=2)      # load_and_play("tictactoe_agent.pt")
    # agent_play_against_heuristics("best_models/tictactoe_agent_0.01879117079079151.pt")
