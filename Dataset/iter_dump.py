import numpy as np
import sys
sys.path.append("..")
from game2048.expectimax import board_to_move
from game2048.game import Game
from game2048.agents import ExpectiMaxAgent
from game2048.displays import Display
import csv, os

GAME_SIZE = 4
SCORE_TO_WIN = 2048
iter_num = 300

game = Game(GAME_SIZE, SCORE_TO_WIN)
board = game.board
agent = ExpectiMaxAgent(game, Display())
direction = agent.step()
board = game.move(direction)

i = 0
dic = {}
idx = 0

# ------------------------------------------------------
# save each board and its direction to a dict
# -------------------------------------------------------
filename = '/home/zhouykai/Workspace/MachinceLearning/Dataset_2048/Train.csv'
# filename = './Dataset/Train.csv'
if os.path.exists(filename):
	head = True
else:
	head = False
	os.mknod(filename)

with open(filename, "a") as csvfile:
	writer = csv.writer(csvfile)
	if not head:
		writer.writerow(["R1C1","R1C2","R1C3","R1C4",\
			"R2C1","R2C2","R2C3","R2C4",\
			"R3C1","R3C2","R3C3","R3C4",\
			"R4C1","R4C2","R4C3","R4C4",\
			"direction"])

	while i < iter_num:

		game = Game(GAME_SIZE, SCORE_TO_WIN)
		board = game.board
		print('Iter idx:', i)

		while(game.end == 0):
			agent = ExpectiMaxAgent(game, Display())
			direction = agent.step()
			board = game.board
			board[board==0] = 1
			board = np.log2(board).flatten().tolist()

			key = str(idx+1)
			idx = idx + 1
			dic[key] = [np.int32(board), np.int32(direction)]
			data = np.int32(np.append(board, direction))
			writer.writerow(data)

			if idx%100 == 0:
				print(data)

			game.move(direction)

		i = i + 1
