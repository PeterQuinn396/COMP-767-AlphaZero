from alphazero import AlphaZero
from tictactoe import tictactoe
import torch
import numpy as np

device = "cuda:0" if torch.cuda.is_available else "cpu"
device = "cpu"
print(f"Using {device}")



def load_and_play(filename, agent_plays=1):
    game = tictactoe()
    input_size = game.obs_space_size
    output_size = game.action_space_size
    hidden_layer_size = 256
    agent = AlphaZero(input_size, hidden_layer_size, output_size)
    agent.load_state_dict(torch.load(filename))

    play_with_agent(agent, verbose=True, agent_plays=agent_plays)

def get_agent_action(agent, game, state, verbose=False):
    mask = game.getLegalActionMask()
    x = torch.tensor(state, device=device, dtype=torch.float)
    y = agent(x)
    pi = y[:-1]
    v = y[-1].detach().numpy()
    _pi = pi.detach().numpy()
    _pi = _pi * mask
    action = np.argmax(_pi)

    if verbose:
        print(f"Value: {v}, Probs: {_pi}, action: {action}")

    return action

def play_with_agent(agent, verbose=False, agent_plays=1):
    game = tictactoe()
    obs, reward, done = game.reset()
    game.render()
    while not done:
        if agent_plays==1:
            agent_action = get_agent_action(agent,game,obs,verbose=verbose)
            obs, reward, done = game.step(agent_action)
            game.render()
            # player takes turn
            if not done:

                action = int(input("Input play space (0-8): "))

                while not game.isLegalAction(action):
                    print("Illegal action")
                    action =  int(input("Input play space (0-8): "))

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
                agent_action = get_agent_action(agent, game, obs, verbose=verbose)
                obs, reward, done = game.step(agent_action)
                game.render()

        else:
            raise Exception("Invalid value for agent plays parameter. Choose 1 or 2")

    if reward == 0:
        print("Tie game")
    elif reward == 1 and agent_plays==1:
        print("Agent won")
    else:
        print("You won!")



if __name__ == "__main__":
   load_and_play("tictactoe_agent_0.14443549513816833.pt", agent_plays=2 ) # works really well
    # load_and_play("tictactoe_agent.pt")