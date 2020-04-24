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
        if action<0:
            return False
        if action>8:
            return False
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

    def get_computer_move(self):  # gets the optimal move for the current player based on some simple heuristics
        # based on the example heuristics found in https://inventwithpython.com/chapter10.html

        # 1. check if the current player can win immediately
        for i in range(9):
            test_game = self.copy()
            if test_game.isLegalAction(i):
                obs, reward, done = test_game.step(i)
                if (done and reward == self.turn):
                    return i

        # 2. Check if the other player has a square where they could win, if so block them

        for i in range(9):
            test_game = self.copy()
            if test_game.isLegalAction(i):
                test_game.game_state[i // 3, i % 3] = -self.turn
                done, outcome = test_game.isGameOver()
                if done:  # other player would win by playing here
                    return i

        # 3. Take the center
        if self.isLegalAction(4):
            return 4


        # 4. Take a corner if it is free
        corners = [0, 2, 6, 8]
        for c in corners:
            if self.isLegalAction(c):
                return c


        # 5. Take a side
        sides = [1, 3, 5, 7]
        for s in sides:
            if self.isLegalAction(s):
                return s


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

