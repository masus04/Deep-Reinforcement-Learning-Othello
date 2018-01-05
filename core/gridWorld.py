import numpy as np
from random import randint

import core.config as config


class GridWorldBoard:

    def __init__(self, board=None, obstacles=15):
        if board is not None:
            self.board = board
        else:
            self.win = None
            self.position = (0, randint(0, 7))

            self.board = np.full((8, 8), config.EMPTY, dtype=np.float64)  # 8 by 8 empty board
            for i in range(obstacles):
                self.board[randint(0, 7)][randint(0, 7)] = 0

            self.board[self.position[0]][self.position[1]] = config.BLACK
            self.board[7][randint(0, 7)] = config.WHITE

    def in_bounds(self, move):
        return move[0] >= 0 and move[0] < 8 and move[1] >= 0 and move[1] < 8

    def get_valid_moves(self, color):
        if color != config.BLACK:
            return []

        moves = []

        for i in range(2):
            for j in range(-1, 2):
                x = self.position[0] + i
                y = self.position[1] + j

                if self.in_bounds((x,y)) and (self.board[x][y] == config.EMPTY or self.board[x][y] == config.WHITE):
                    moves.append((x, y))

        return moves

    def apply_move(self, move, color):
        self.board[move[0]][move[1]] = config.BLACK
        self.position = move
        return self

    def game_won(self):
        if not (self.board == np.full((8, 8), config.WHITE, dtype=np.float64)).any():
            self.win = config.BLACK

        elif len(self.get_valid_moves(config.BLACK)) == 0:
            self.win = config.WHITE

        return self.win

    def get_representation(self, color):
        return np.copy(self.board)

    def copy(self):
        return GridWorldBoard(np.copy(self.board))

