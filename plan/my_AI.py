import numpy as np
import heapq
import copy


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
	# print(core)
	return core


def move(board, direction):
	board_to_left = np.rot90(board, -direction)
	for row in range(4):
		core = merge(board_to_left[row])
		board_to_left[row, :len(core)] = core
		board_to_left[row, len(core):] = 0
	board = np.rot90(board_to_left, direction)

	where_empty = list(zip(*np.where(board == 0)))
	if where_empty:
		selected = where_empty[np.random.randint(0, len(where_empty))]
		board[selected] = 2 if np.random.random() < 0.5 else 4

	return board


def evaluate_board(origin_board):
	ratio = 0.2
	weight = 1.0
	weighted_value = 0

	max_index = np.where(origin_board == np.max(origin_board))
	flag1 = max_index[0]%3 == 0
	flag2 = max_index[1]%3 == 0

	if not (flag1.all() and flag2.all()):
		return 0, [0, 0]

	for direction in range(4):

		board = np.rot90(origin_board, direction)
		# print(board)
		current_value = 0
		for i in range(7):
			for j in range(i+1):
				if (i-j <= 3 and j<=3):
					current_value = current_value + board[j, i-j]*weight
			weight = weight*ratio

		# print(current_weight)
		if(current_value >= weighted_value):
			weighted_value = current_value
	# print(weighted_value)
	
	where_empty = list(zip(*np.where(board == 0)))
	if where_empty:
		selected = where_empty[np.random.randint(0, len(where_empty))]
	else:
		selected = [0, 0]
	return weighted_value, selected


def evaluate_board_2(origin_board):
	ratio = 0.2
	weight = 1.0
	weighted_value = 0

	'''
	max_index = np.where(origin_board == np.max(origin_board))
	flag1 = max_index[0]%3 == 0
	flag2 = max_index[1]%3 == 0
	

	if not (flag1.all() and flag2.all()):
		return 0, [0, 0]
	'''

	for direction in range(4):

		board = np.rot90(origin_board, direction)
		# print(board)
		row_value = 0
		col_value = 0
		for i in range(4):
			for j in range(4):
				row_value = row_value + board[i, j]*weight
				col_value = col_value + board[j, i]*weight
				weight = weight*ratio

		if(row_value >= weighted_value):
			weighted_value = row_value
		if(col_value >= weighted_value):
			weighted_value = col_value

	# print(weighted_value)

	where_empty = list(zip(*np.where(board == 0)))
	if where_empty:
		selected = where_empty[np.random.randint(0, len(where_empty))]
	else:
		selected = [0, 0]
	return weighted_value, selected


def recursion_move(board, depth, max_depth):
	best_score = 0
	best_move = 0
	for i in range(4):
		new_board = copy.deepcopy(board)
		new_board = move(new_board, i)

		# score, position = evaluate_board(new_board)
		score, position = evaluate_board_2(new_board)
		if depth != 0:
			next_Move, next_score = recursion_move(new_board, depth-1, max_depth)
			score = score + next_score*pow(0.9, max_depth - depth + 1)

		if(score > best_score):
			best_move = i
			best_score = score

	# print(best_score)
	return best_move, best_score


def get_best_move(board):
	direction, score = recursion_move(board, 3, 3)
	return direction

