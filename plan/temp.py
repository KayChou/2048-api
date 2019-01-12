import numpy as np
from my_AI import evaluate_board, move, recursion_move

# a = np.array([[4,2,2,4], [2,2,2,2], [0,4,4,0], [0,4,4,0]])
a = np.array([[2,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,10]])
print(a, '\n')
value = evaluate_board(a)

new_board = move(a, 3)
print(new_board)
print(value)