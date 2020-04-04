import numpy as np

class tictactoe():

    def __init__(self):
        self.game_state = np.zeros((3, 3))
        self.turn = 1
        self.action_space_size = 9  # 9 places to play
        self.obs_space_size = 9 + 1  # 9 spaces plus player whose turn it is

    def reset(self):
        self.game_state = np.zeros((3, 3))
        self.turn = 1
        obs = self.game_state.flatten()
        obs = np.append(obs, [self.turn])
        obs = obs.astype(np.float32)
        return obs, 0, False

    def step(self, action):
        # action should be a number in range 0-8
        if not self.isLegalAction(action):
            print('Illegal action played')
            print(f"action: {action}")
            self.render()
            print("")
            raise Exception("Illegal action")

        i = action // 3
        j = action % 3
        self.game_state[i, j] = self.turn  # put 1 or -1 in the specified spot
        self.turn = -self.turn  # other players turn
        done, outcome = self.isGameOver()

        if done:
            obs = None  # if we are done, obs should not matter and should not be checked
            return obs, outcome, done

        obs = self.game_state.flatten()
        obs = np.append(obs, [self.turn])
        reward = 0
        done = False

        return obs, reward, done

    def render(self):
        print(self.game_state)

    def isGameOver(self):
        done = False
        # check rows
        for i in range(3):
            same = True
            for j in range(1, 3):
                same = same and self.game_state[i, 0] == self.game_state[i, j]
            if same and self.game_state[i, 0] != 0:
                done = True
                outcome = self.game_state[i, 0]
                return done, outcome

        # check columns
        for j in range(3):
            same = True
            for i in range(1, 3):
                same = same and self.game_state[0, j] == self.game_state[i, j]
            if same and self.game_state[0, j] != 0:
                done = True
                outcome = self.game_state[0, j]
                return done, outcome

        # check diagonal top left - bottom right
        same = True
        for i in range(1, 3):
            j = i
            same = same and self.game_state[0, 0] == self.game_state[i, j]

        if same and self.game_state[0, 0] != 0:
            done = True
            outcome = self.game_state[0, 0]
            return done, outcome

        # check diagonal top right - bottom
        same = True
        for i in range(1, 3):
            j = 2 - i
            same = same and self.game_state[0, 2] == self.game_state[i, j]

        if same and self.game_state[2, 0] != 0:
            done = True
            outcome = self.game_state[2, 0]
            return done, outcome

        # if the whole board is full, the game is over
        # check at least one space has a zero (open to play)
        for i in range(3):
            for j in range(3):
                if self.game_state[i, j] == 0:
                    return False, 0
        # board was full, but nobody won
        return True, 0

    def isLegalAction(self, action):
        i = action // 3
        j = action % 3
        return self.game_state[i, j] == 0

    def getLegalActionMask(self):
        mask = np.zeros(9)
        for i in range(9):
            mask[i] = 1 if self.isLegalAction(i) else 0
        return mask

    def copy(self):
        copy_game = tictactoe()
        copy_game.game_state = np.copy(self.game_state)
        copy_game.turn = self.turn
        return copy_game


def test_tictactoe():
    game = tictactoe()
    obs, reward, done = game.reset()
    game.render()
    print(f"obs: {obs}, reward: {reward}, done: {done}")
    for i in range(8):
        obs, reward, done = game.step(i)
        game.render()
        print(f"obs: {obs}, reward: {reward}, done: {done}")
        if done:
            break

