import numpy as np

def merge(row):
	non_zero = row[row != 0]  # remove zeros
	core = [None]
	for elem in non_zero:
		if core[-1] is None:
			core[-1] = elem
		elif core[-1] == elem:
			core[-1] = 2 * elem
			core.append(None)
		else:
			core.append(elem)
	if core[-1] is None:
		core.pop()
	return core


def empty_num(board):
	num = np.sum(board==0)
	return num


def move(board, direction):
	board_to_left = np.rot90(board, -direction)
	for row in range(4):
		core = merge(board_to_left[row])
		board_to_left[row, :len(core)] = core
		board_to_left[row, len(core):] = 0
	board = np.rot90(board_to_left, direction)
	return board


def evaluate_board(board):
	point1 = empty_num(board)
	final_point = point1
	return final_point


def get_best_move(board, depth):
	score = np.zeros([depth, 4])

	for i in range(depth):
		for direction in range(4):
			next_board = move(board, direction)
			print(next_board)
			score[i, direction] = evaluate_board(next_board)

	return score