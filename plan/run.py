import numpy as np
import sys
sys.path.append("..")
from game2048.expectimax import board_to_move
from game2048.game import Game
from game2048.agents import ExpectiMaxAgent
from game2048.displays import Display
import csv, os
from utils import *

GAME_SIZE = 4
SCORE_TO_WIN = 2048
iter_num = 50000

game = Game(GAME_SIZE, SCORE_TO_WIN)
board = game.board
agent = ExpectiMaxAgent(game, Display())
direction = agent.step()
board = game.move(direction)

agent = ExpectiMaxAgent(game, Display())
direction = agent.step()
board = game.board
# board[board==0] = 1
# board = np.log2(board).flatten().tolist()
print(board, direction)
next_board = get_best_move(board, 4)
print(next_board)
'''
game.move(direction)
direction = agent.step()
board = game.board
print(board, direction)

'''