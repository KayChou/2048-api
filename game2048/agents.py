import numpy as np
import os
import sys
import torch
import pickle
import joblib
from torchvision import transforms

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction


class randomForest(Agent):

    def __init__(self, game, display=None):

        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)

        path = os.getcwd()
        model_path = path + "/Dataset/model.joblib"
        print("Loading Model:")
        clf = joblib.load(open(model_path, 'rb'))
        print("Model Loaded")

        self.search_func = clf.predict

    def step(self):
        board = self.game.board
        board[board==0] = 1
        board = np.log2(board).flatten().tolist()
        
        tmp = np.zeros([1, 16])
        tmp[0, :] = board
        direction = self.search_func(tmp)
        return int(direction)


class CNN(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        path = os.getcwd()
        model_path = path + '/Dataset/model.pth'
        print("Loading Model:")
        model = torch.load(model_path, map_location='cpu')
        print("Model Loaded")

        self.search_func = model

    def pre_process(self):
        board = self.game.board
        board[board==0] = 1
        board = np.log2(board).flatten()

        board = board.reshape((4, 4))
        board = board[:, :, np.newaxis]
        board = board/11.0

        board = transforms.ToTensor()(board)
        # board = transforms.Normalize((0.5,), (0.3081,))(board)
        board = torch.unsqueeze(board, dim=0).float()
        board = board.repeat(1, 1, 1, 1)
        return board

    def get_direction(self, model_output):
        max_idx = np.where(model_output==torch.max(model_output))[1]
        counts = np.bincount(max_idx)
        d = np.argmax(counts)
        return d

    def step(self):
        input = self.pre_process()
        output = self.search_func(input)
        direction = self.get_direction(output)
        return int(direction)
