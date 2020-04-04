from alphazero import AlphaZero
from tictactoe import tictactoe
import torch
import numpy as np

device = "cuda:0" if torch.cuda.is_available else "cpu"
device = "cpu"
print(f"Using {device}")



def load_and_play():
    game = tictactoe()
    input_size = game.obs_space_size
    output_size = game.action_space_size
    hidden_layer_size = 256
    agent = AlphaZero(input_size, hidden_layer_size, output_size)
    agent.load_state_dict(torch.load("tictactoe_agent.pt"))

    play_with_agent(agent, verbose=True)


def play_with_agent(agent, verbose=False):
    game = tictactoe()
    obs, reward, done = game.reset()
    agent_plays = 1  # agent plays first
    game.render()
    while not done:

        mask = game.getLegalActionMask()
        x = torch.tensor(obs, device=device, dtype=torch.float)
        y = agent(x)
        pi = y[:-1]
        v = y[-1].detach().numpy()
        _pi = pi.detach().numpy()
        _pi = _pi * mask
        action = np.argmax(_pi)

        if verbose:
            print(f"Value: {v}, Probs: {_pi}, action: {action}")

        obs, reward, done = game.step(action)
        game.render()
        # player takes turn
        if not done:

            action = int(input("Input play space (0-8): "))

            while not game.isLegalAction(action):
                print("Illegal action")
                action =  int(input("Input play space (0-8): "))

            obs, reward, done = game.step(action)
            game.render()

    if reward == 1:
        print("Agent won")
    elif reward == -1:
        print("You won!")
    else:
        print("Tie game")


if __name__ == "__main__":
    load_and_play()