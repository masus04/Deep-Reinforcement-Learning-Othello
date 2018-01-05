import numpy as np
import core.config as config
from random import randint


class GridWorldBoard:

    def __init__(self, board=None):
        if board:
            self.board = board
        else:
            self.moves = 0
            self.position = (0, randint(9))

            self.board = np.full((8, 8), config.EMPTY, dtype=np.float64)  # 8 by 8 empty board
            self.board[self.position[0]][self.position[1]] = config.BLACK
            self.board[8][randint(9)] = config.WHITE
            for i in range(15):
                self.board[randint(9)][randint(9)] = 0

    def get_valid_moves(self, color):
        if color != config.BLACK:
            return []

        moves = []

        for i in range(-1, 2):
            for j in range(-1, 2):
                x = self.position[0] + i
                y = self.position[1] + j
                try:
                    if self.board[x][y] == config.EMPTY:
                        moves.append((x, y))
                except IndexError:
                    pass

    def apply_move(self, move, color):
        if color != config.BLACK:
            return

        self.board[move[0]][move[1]] = config.BLACK

    def game_won(self):
        if self.board[self.position[0]][self.position[1]] == config.BLACK:
            return config.BLACK

        if len(self.get_valid_moves()):
            return config.WHITE

        return None

    def get_representation(self, color):
        return np.copy(self.board)

    def copy(self):
        return GridWorldBoard(np.copy(self.board))
