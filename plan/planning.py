import numpy as np
import heapq
from plan.utils import AI, Board

'''
array = np.array([[0,0,0,2], [0,0,0,4], [0,0,0,4],[0,0,0,16]])
a = Board()
a.setBoard(array)
print(a.getLine(0), a.getLine(1), a.getLine(2), a.getLine(3))
model = AI()
direction = model.nextMove(a, 3)
print(direction)
'''

def get_next_move(array):
	a = Board()
	a.setBoard(array)

	model = AI()
	direction = model.nextMove(a, 3)
	return int(direction)
