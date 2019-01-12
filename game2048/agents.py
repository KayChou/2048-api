import numpy as np
import os, time
import sys
import torch
import pickle
import joblib
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
import heapq
sys.path.append("..")
from plan.my_AI import get_best_move


tmp = np.array(range(12))
tmp = tmp[:, np.newaxis]
fit_array = tmp.repeat(16, 1)
fit_array = np.array(fit_array)

enc = OneHotEncoder(n_values = 'auto')
enc.fit(fit_array)


def one_hot(board):

    board = board[np.newaxis, :]
    board = enc.transform(board)
    board = board.A.reshape(4, 4, 12)
    return board


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


class CNN(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        path = os.getcwd()
        model_path = path + '/model.pth'
        print("Loading Model:")
        model = torch.load(model_path, map_location='cpu')
        print("Model Loaded")

        self.search_func = model

    def pre_process(self):
        board = self.game.board
        board[board==0] = 1
        board = np.log2(board)
        board = board[:, :, np.newaxis]
        board = board/11.0

        board = transforms.ToTensor()(board)
        # board = transforms.Normalize((0.5,), (0.3081,))(board)
        board = torch.unsqueeze(board, dim=0).float()
        board = board.repeat(64, 1, 1, 1)
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


class oneHot_CNN(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        path = os.getcwd()
        model_path = path + '/model.pth'
        print("Loading Model:")
        model = torch.load(model_path, map_location='cpu')
        print("Model Loaded")

        self.search_func = model

    def pre_process(self):
        board = self.game.board
        board[board==0] = 1
        board = np.log2(board).flatten()
        board = one_hot(board)
        board = board/1.0

        board = transforms.ToTensor()(board)
        # board = transforms.Normalize((0.5,), (0.3081,))(board)
        board = torch.unsqueeze(board, dim=0).float()
        board = board.repeat(64, 1, 1, 1)
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


class RNN_model(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        path = os.getcwd()
        model_path = path + '/model.pth'
        # model1_path = path + '/model1.pth'
        print("Loading Model:")
        self.model = torch.load(model_path, map_location='cpu')
        # self.model1 = torch.load(model_path, map_location='cpu')
        print("Model Loaded")

        self.search_func = self.model

    def pre_process(self):
        board = self.game.board
        '''
        if np.max(board) < 8:
            self.search_func = self.model1
        else:
            self.search_func = self.model
        '''
        self.search_func = self.model
        
        board[board==0] = 1
        board = np.log2(board)
        board = board[:, :, np.newaxis]
        board = board/11.0

        board = transforms.ToTensor()(board)
        return board.float()

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


class RNN_rot(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        path = os.getcwd()
        model_path = path + '/model.pth'
        # model1_path = path + '/model1.pth'
        print("Loading Model:")
        self.model = torch.load(model_path, map_location='cpu')
        self.model1 = torch.load(path + '/model_back.pth', map_location='cpu')
        print("Model Loaded")

        self.search_func = self.model

    def pre_process(self, i):
        
        board = self.game.board
        board = np.rot90(board, i)
        board[board==0] = 1
        board = np.log2(board)
        board = board[:, :, np.newaxis]
        board = board/11.0

        board = transforms.ToTensor()(board)
        return board.float()

    def get_direction(self, model_output):
        max_idx = np.where(model_output==torch.max(model_output))[1]
        counts = np.bincount(max_idx)
        d = np.argmax(counts)
        return d

    def step(self):
        # start_time = time.time()
        directions = np.zeros([4])
        for i in range(4):
            input = self.pre_process(i)
            output = self.search_func(input)
            directions[i] = (self.get_direction(output) + 4 - i)%4
        # print(directions)
        counts = np.bincount(directions.astype(int))
        top2 = heapq.nlargest(2, counts)
        new_counts = counts

        # use second model
        if(len(top2)>=2):
            if(top2[0] == top2[1]): 
                new_directions = np.zeros([4])
                for i in range(4):
                    input = self.pre_process(i)
                    output = self.model1(input)
                    new_directions[i] = (self.get_direction(output) + 4 - i)%4
                # print(directions)
                new_counts = np.bincount(new_directions.astype(int))
                # new_counts = np.append(counts, new_counts)

        direction = np.argmax(new_counts)

        # print("Time used:", time.time()-start_time)
        return int(direction)


class RNN_layer(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        path = os.getcwd()
        model_path = path + '/model.pth'
        # model1_path = path + '/model1.pth'
        print("Loading Model:")
        self.model512 = torch.load(path+'/model512.pth', map_location='cpu')
        self.model1024 = torch.load(path+'/model512.pth', map_location='cpu')
        self.model2048 = torch.load(path+'/model512.pth', map_location='cpu')
        print("Model Loaded")

        # self.search_func = self.model

    def pre_process(self):
        board = self.game.board

        if np.max(board) <= 8:
            self.search_func = self.model512
        if np.max(board) == 9:
            self.search_func = self.model1024
        if np.max(board) >= 10:
            self.search_func = self.model2048

        board[board==0] = 1
        board = np.log2(board)
        board = board[:, :, np.newaxis]
        board = board/11.0

        board = transforms.ToTensor()(board)
        return board.float()

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

class my_AI(Agent):
    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)

        self.search_func = get_best_move

    def step(self):
        start_time = time.time()
        direction = self.search_func(self.game.board)
        # print("Time used:", time.time()-start_time)
        return direction