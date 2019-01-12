import random
import copy
import numpy as np
from plan.my_AI import evaluate_board

# -*- coding: UTF-8 -*-

# PY3 compat
try:
    xrange
except NameError:
    xrange = range


class Board(object):
    """
    A 2048 board
    """

    # UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4
    UP, DOWN, LEFT, RIGHT = 3, 1, 0, 2

    GOAL = 204800
    SIZE = 4

    def __init__(self, goal=GOAL, size=SIZE, **kws):
        self.__size = size
        self.__size_range = xrange(0, self.__size)
        self.__goal = goal
        self.__won = False
        self.cells = [[0]*self.__size for _ in xrange(self.__size)]
        self.addTile()
        self.addTile()

    def size(self):
        """return the board size"""
        return self.__size

    def goal(self):
        """return the board goal"""
        return self.__goal

    def won(self):
        """
        return True if the board contains at least one tile with the board goal
        """
        return self.__won

    def maxValue(self):
        """
        return the max value in the board
        """
        maxVal = 0
        for y in self.__size_range:
            for x in self.__size_range:
                maxVal = max(self.getCell(x,y),maxVal)
        return maxVal

    def canMove(self):
        """
        test if a move is possible
        """
        if not self.filled():
            return True

        for y in self.__size_range:
            for x in self.__size_range:
                c = self.getCell(x, y)
                if (x < self.__size-1 and c == self.getCell(x+1, y)) \
                        or (y < self.__size-1 and c == self.getCell(x, y+1)):
                    return True

        return False

    def validMove(self, dir):
        """
        test if a move is possible
        """
        if dir == self.UP or dir == self.DOWN:
            for x in self.__size_range:
                col = self.getCol(x)
                for y in self.__size_range:
                    if(y < self.__size-1 and col[y] == col[y+1] and col[y]!=0):
                        return True
                    if(dir == self.DOWN and y > 0 and col[y] == 0 and col[y-1]!=0):
                        return True
                    if(dir == self.UP and y < self.__size-1 and col[y] == 0 and col[y+1]!=0):
                        return True        
        
        if dir == self.LEFT or dir == self.RIGHT:
            for y in self.__size_range:
                line = self.getLine(y)
                for x in self.__size_range:
                    if(x < self.__size-1 and line[x] == line[x+1] and line[x]!=0):
                        return True
                    if(dir == self.RIGHT and x > 0 and line[x] == 0 and line[x-1]!=0):
                        return True
                    if(dir == self.LEFT and x < self.__size-1 and line[x] == 0 and line[x+1]!=0):
                        return True        
        return False

    def filled(self):
        """
        return true if the game is filled
        """
        return len(self.getEmptyCells()) == 0

    def addTile(self, value=None, choices=([2]*9+[4])):
        """
        add a random tile in an empty cell
          value: value of the tile to add.
          choices: a list of possible choices for the value of the tile.
                   default is [2, 2, 2, 2, 2, 2, 2, 2, 2, 4].
        """
        if value:
            choices = [value]

        v = random.choice(choices)
        empty = self.getEmptyCells()
        if empty:
            x, y = random.choice(empty)
            self.setCell(x, y, v)

    def getCell(self, x, y):
        """return the cell value at x,y"""
        return self.cells[y][x]

    def setCell(self, x, y, v):
        """set the cell value at x,y"""
        self.cells[y][x] = v

    def getLine(self, y):
        """return the y-th line, starting at 0"""
        return self.cells[y]

    def getCol(self, x):
        """return the x-th column, starting at 0"""
        return [self.getCell(x, i) for i in self.__size_range]

    def setLine(self, y, l):
        """set the y-th line, starting at 0"""
        self.cells[y] = l[:]

    def setCol(self, x, l):
        """set the x-th column, starting at 0"""
        for i in xrange(0, self.__size):
            self.setCell(x, i, l[i])

    def setBoard(self, array):
        for i in range(0, 4):
            for j in range(0, 4):
                self.setCell(i, j, array[j, i])

    def getEmptyCells(self):
        """return a (x, y) pair for each empty cell"""
        return [(x, y) for x in self.__size_range
                           for y in self.__size_range if self.getCell(x, y) == 0]

    def __collapseLineOrCol(self, line, d):
        """
        Merge tiles in a line or column according to a direction and return a
        tuple with the new line and the score for the move on this line
        """
        if (d == Board.LEFT or d == Board.UP):
            inc = 1
            rg = xrange(0, self.__size-1, inc)
        else:
            inc = -1
            rg = xrange(self.__size-1, 0, inc)

        pts = 0
        for i in rg:
            if line[i] == 0:
                continue
            if line[i] == line[i+inc]:
                v = line[i]*2
                if v == self.__goal:
                    self.__won = True

                line[i] = v
                line[i+inc] = 0
                pts += v

        return (line, pts)

    def __moveLineOrCol(self, line, d):
        """
        Move a line or column to a given direction (d)
        """
        nl = [c for c in line if c != 0]
        if d == Board.UP or d == Board.LEFT:
            return nl + [0] * (self.__size - len(nl))
        return [0] * (self.__size - len(nl)) + nl

    def move(self, d, add_tile=True):
        """
        move and return the move score
        """
        if d == Board.LEFT or d == Board.RIGHT:
            chg, get = self.setLine, self.getLine
        elif d == Board.UP or d == Board.DOWN:
            chg, get = self.setCol, self.getCol
        else:
            return 0

        moved = False
        score = 0

        for i in self.__size_range:
            # save the original line/col
            origin = get(i)
            # move it
            line = self.__moveLineOrCol(origin, d)
            # merge adjacent tiles
            collapsed, pts = self.__collapseLineOrCol(line, d)
            # move it again (for when tiles are merged, because empty cells are
            # inserted in the middle of the line/col)
            new = self.__moveLineOrCol(collapsed, d)
            # set it back in the board
            chg(i, new)
            # did it change?
            if origin != new:
                moved = True
            score += pts

        # don't add a new tile if nothing changed
        if moved and add_tile:
            self.addTile()

        return score


class AI(object):
    def __str__(self, margins={}):
        return ""
        

    @staticmethod
    def randomNextMove(board):
        '''
        It's just a test for the validMove function
        '''
        if board.validMove(Board.UP):
            print ("UP: ok")
        else:
            print ("UP: no")
        if board.validMove(Board.DOWN):
            print ("DOWN: ok")
        else:
            print ("DOWN: no")
        if board.validMove(Board.LEFT):
            print ("LEFT: ok")
        else:
            print ("LEFT: no")
        if board.validMove(Board.RIGHT):
            print ("RIGHT: ok")
        else:
            print ("RIGHT: no")
        rm = random.randrange(0, 4)
        print (rm) 
        return rm
    
    @staticmethod
    def nextMove(board, recursion_depth=3):
        m, s = AI.nextMoveRecur(board, recursion_depth, recursion_depth)
        return m
        
    @staticmethod
    def nextMoveRecur(board, depth, maxDepth, base=0.9):
        bestScore = -1.
        bestMove = 0
        for m in range(4):
            if(board.validMove(m)):
                newBoard = copy.deepcopy(board)
                newBoard.move(m, add_tile=False)
                
                # score, critical = AI.evaluate(newBoard)
                score, critical = evaluate_board(newBoard.cells)
                newBoard.setCell(critical[0], critical[1], 2)
                if depth != 0:
                    my_m, my_s = AI.nextMoveRecur(newBoard, depth-1, maxDepth)
                    score += my_s*pow(base, maxDepth - depth + 1)
                
                if(score > bestScore):
                    bestMove = m
                    bestScore = score
        # print(bestScore)
        return (bestMove, bestScore);

    #Hey!!! Don't judge me for this awful piece of code!!!
    #It's just a quick test...
    '''
    @staticmethod
    
    def evaluate(board, commonRatio=0.25):
        linearWeightedVal = 0
        invert = False
        weight = 1.
        malus = 0
        criticalTile = (-1,-1)
        for y in range(0, board.size()):
            for x in range(0, board.size()):
                b_x = x
                b_y = y
                if invert:
                    b_x = board.size() - 1 - x
                #linearW
                currVal = board.getCell(b_x, b_y)
                if(currVal == 0 and criticalTile == (-1,-1)):
                    criticalTile = (b_x, b_y)
                linearWeightedVal = linearWeightedVal + currVal*weight
                weight = weight*commonRatio
            invert = not invert
        
        linearWeightedVal2 = 0
        invert = False
        weight = 1.
        malus = 0
        criticalTile2 = (-1,-1)
        for x in range(0,board.size()):
            for y in range(0,board.size()):
                b_x = x
                b_y = y
                if invert:
                    b_y = board.size() - 1 - y
                #linearW
                currVal=board.getCell(b_x,b_y)
                if(currVal == 0 and criticalTile2 == (-1,-1)):
                    criticalTile2 = (b_x,b_y)
                linearWeightedVal2 += currVal*weight
                weight *= commonRatio
            invert = not invert
            
        
        linearWeightedVal3 = 0
        invert = False
        weight = 1.
        malus = 0
        criticalTile3 = (-1,-1)
        for y in range(0,board.size()):
            for x in range(0,board.size()):
                b_x = x
                b_y = board.size() - 1 - y
                if invert:
                    b_x = board.size() - 1 - x
                #linearW
                currVal=board.getCell(b_x,b_y)
                if(currVal == 0 and criticalTile3 == (-1,-1)):
                    criticalTile3 = (b_x,b_y)
                linearWeightedVal3 += currVal*weight
                weight *= commonRatio
            invert = not invert
            
        linearWeightedVal4 = 0
        invert = False
        weight = 1.
        malus = 0
        criticalTile4 = (-1,-1)
        for x in range(0,board.size()):
            for y in range(0,board.size()):
                b_x = board.size() - 1 - x
                b_y = y
                if invert:
                    b_y = board.size() - 1 - y
                #linearW
                currVal=board.getCell(b_x,b_y)
                if(currVal == 0 and criticalTile4 == (-1, -1)):
                    criticalTile4 = (b_x,b_y)
                linearWeightedVal4 += currVal*weight
                weight *= commonRatio
            invert = not invert
            
            
        linearWeightedVal5 = 0
        invert = True
        weight = 1.
        malus = 0
        criticalTile5 = (-1,-1)
        for y in range(0,board.size()):
            for x in range(0,board.size()):
                b_x = x
                b_y = y
                if invert:
                    b_x = board.size() - 1 - x
                #linearW
                currVal=board.getCell(b_x,b_y)
                if(currVal == 0 and criticalTile5 == (-1,-1)):
                    criticalTile5 = (b_x,b_y)
                linearWeightedVal5 += currVal*weight
                weight *= commonRatio
            invert = not invert
            
        linearWeightedVal6 = 0
        invert = True
        weight = 1.
        malus = 0
        criticalTile6 = (-1,-1)
        for x in range(0,board.size()):
            for y in range(0,board.size()):
                b_x = x
                b_y = y
                if invert:
                    b_y = board.size() - 1 - y
                #linearW
                currVal=board.getCell(b_x,b_y)
                if(currVal == 0 and criticalTile6 == (-1,-1)):
                    criticalTile6 = (b_x,b_y)
                linearWeightedVal6 += currVal*weight
                weight *= commonRatio
            invert = not invert
            
        
        linearWeightedVal7 = 0
        invert = True
        weight = 1.
        malus = 0
        criticalTile7 = (-1,-1)
        for y in range(0,board.size()):
            for x in range(0,board.size()):
                b_x = x
                b_y = board.size() - 1 - y
                if invert:
                    b_x = board.size() - 1 - x
                #linearW
                currVal=board.getCell(b_x,b_y)
                if(currVal == 0 and criticalTile7 == (-1,-1)):
                    criticalTile7 = (b_x,b_y)
                linearWeightedVal7 += currVal*weight
                weight *= commonRatio
            invert = not invert
            
        linearWeightedVal8 = 0
        invert = True
        weight = 1.
        malus = 0
        criticalTile8 = (-1,-1)
        for x in range(0,board.size()):
            for y in range(0,board.size()):
                b_x = board.size() - 1 - x
                b_y = y
                if invert:
                    b_y = board.size() - 1 - y
                #linearW
                currVal=board.getCell(b_x,b_y)
                if(currVal == 0 and criticalTile8 == (-1,-1)):
                    criticalTile8 = (b_x,b_y)
                linearWeightedVal8 += currVal*weight
                weight *= commonRatio
            invert = not invert
            
        maxVal = max(linearWeightedVal,linearWeightedVal2,linearWeightedVal3,\
            linearWeightedVal4,linearWeightedVal5,linearWeightedVal6,\
            linearWeightedVal7,linearWeightedVal8)
        if(linearWeightedVal2 > linearWeightedVal):
            linearWeightedVal = linearWeightedVal2
            criticalTile = criticalTile2
        if(linearWeightedVal3 > linearWeightedVal):
            linearWeightedVal = linearWeightedVal3
            criticalTile = criticalTile3
        if(linearWeightedVal4 > linearWeightedVal):
            linearWeightedVal = linearWeightedVal4
            criticalTile = criticalTile4
        if(linearWeightedVal5 > linearWeightedVal):
            linearWeightedVal = linearWeightedVal5
            criticalTile = criticalTile5
        if(linearWeightedVal6 > linearWeightedVal):
            linearWeightedVal = linearWeightedVal6
            criticalTile = criticalTile6
        if(linearWeightedVal7 > linearWeightedVal):
            linearWeightedVal = linearWeightedVal7
            criticalTile = criticalTile7
        if(linearWeightedVal8 > linearWeightedVal):
            linearWeightedVal = linearWeightedVal8
            criticalTile = criticalTile8
        
        return maxVal, criticalTile
    '''