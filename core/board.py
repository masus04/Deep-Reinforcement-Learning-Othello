from core.config import EMPTY, BLACK, WHITE, other_color
import numpy as np
from ctypes import *
import random


class Board:

    LIBFUNCTIONS = cdll.LoadLibrary("./core/libfunctions.so")
    BOARD_SIZE = 8

    def __init__(self, board = None):
        if board is not None:
            self.board = board
            self.count_pieces()
        else:
            self.board = np.full((8, 8), EMPTY, dtype=np.float64)  # 8 by 8 empty board
            self.board[3][4] = BLACK
            self.board[4][3] = BLACK
            self.board[3][3] = WHITE
            self.board[4][4] = WHITE
            self.white_pieces = 2
            self.black_pieces = 2
            self.empty_spaces = 60
        self.valid_moves = []
        self.now_playing = BLACK

    def get_valid_moves(self, color):
        v = Board.LIBFUNCTIONS.get_valid_moves(c_void_p(self.board.ctypes.data), color)
        c_int_p_p = POINTER(POINTER(c_int))
        # print(v, c_int_p_p)
        moves = cast(v, c_int_p_p)
        valid_moves = [None] * moves[0][0]
        for i in range(moves[0][0]):
            valid_moves[i] = (moves[i + 1][0], moves[i + 1][1])
        self.valid_moves = valid_moves
        Board.LIBFUNCTIONS.free_moves(v, moves[0][0])
        # ----------------------------------------------------------
        python_moves = self.get_valid_moves_python(color)
        if not (set(valid_moves) == set(python_moves)):
            print(valid_moves, python_moves, set(valid_moves) == (python_moves))
        # ----------------------------------------------------------
        return valid_moves

    def get_valid_moves_python(self, color):
        if color == BLACK:
            num_pieces = self.black_pieces
        else:
            num_pieces = self.white_pieces
        if num_pieces < self.empty_spaces:
            return self.get_valid_moves_occupied(color)
        else:
            return self.get_valid_moves_empty(color)

    def get_valid_moves_occupied(self, color):
        directions = []
        for i in range(3):
            for j in range(3):
                if not (i == 1 and j == 1):
                    directions.append((i-1, j-1))

        moves = set()
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if self.board[i][j] == color:
                    for dir in directions:
                        candidate = self.search_valid_moves_occupied((i, j), dir, color)
                        if candidate:
                            moves.add(candidate)
        return list(moves)

    def search_valid_moves_occupied(self, position, direction, color, depth=0):
        i = position[0] + direction[0]
        j = position[1] + direction[1]

        if not self.in_bouds(i, j) or self.board[i][j] == color:
            return None

        if self.board[i][j] == EMPTY and depth != 0:
            return i, j

        if self.board[i][j] == other_color(color):
            return self.search_valid_moves_occupied((i, j), direction, color, depth=depth+1)

    def get_valid_moves_empty(self, color):
        directions = []
        for i in range(3):
            for j in range(3):
                if not (i == 1 and j == 1):
                    directions.append((i - 1, j - 1))

        moves = set()
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if self.board[i][j] == EMPTY:
                    for dir in directions:
                        if self.search_valid_moves_empty((i, j), dir, color):
                            moves.add((i, j))
        return list(moves)

    def search_valid_moves_empty(self, position, direction, color, depth=0):
        i = position[0] + direction[0]
        j = position[1] + direction[1]

        if not self.in_bouds(i, j) or self.board[i][j] == EMPTY:
            return False

        if self.board[i][j] == color and depth > 0:
            return True

        if self.board[i][j] == other_color(color):
            return self.search_valid_moves_empty((i, j), direction, color, depth=depth+1)

    @staticmethod
    def in_bouds(i, j):
        return i >= 0 and i < 8 and j >= 0 and j < 8

    def apply_move(self, move, color):
        self.board[move[0]][move[1]] = color
        if color == BLACK:
            self.black_pieces += 1
        else:
            self.white_pieces += 1
        self.empty_spaces -= 1
        self.flip_pieces(move, color)
        return self

    def flip_pieces(self, position, color):
        for direction in range(1,9): # Flip row for each of the 8 possible directions
            (num_pieces, pieces_to_flip) = self.pieces_to_flip_in_row(position, color, direction)
            for i in range(num_pieces):
                self.board[pieces_to_flip[i][0]][pieces_to_flip[i][1]] = color
            if color == BLACK:
                self.black_pieces += num_pieces
                self.white_pieces -= num_pieces
            else:
                self.black_pieces -= num_pieces
                self.white_pieces += num_pieces

    def pieces_to_flip_in_row(self, position, color, direction):
        row_inc = 0
        col_inc = 0
        if direction >= 5:
            direction += 1 # Have directions correspond to numberpad
        if direction == 1 or direction == 2 or direction == 3:
            row_inc = -1
        elif direction == 7 or direction == 8 or direction == 9:
            row_inc = 1
        if direction == 1 or direction == 4 or direction == 7:
            col_inc = -1
        elif direction == 3 or direction == 6 or direction == 9:
            col_inc = 1

        pieces = [None] * 8
        pieces_flipped = 0
        i = position[0] + row_inc
        j = position[1] + col_inc

        if color == WHITE:
            other = BLACK
        else:
            other = WHITE

        if i in range(8) and j in range(8) and self.board[i][j] == other:
            # assures there is at least one piece to flip
            pieces[pieces_flipped] = (i,j)
            pieces_flipped += 1
            i = i + row_inc
            j = j + col_inc
            while i in range(8) and j in range(8) and self.board[i][j] == other:
                # search for more pieces to flip
                pieces[pieces_flipped] = (i,j)
                pieces_flipped += 1
                i = i + row_inc
                j = j + col_inc
            if i in range(8) and j in range(8) and self.board[i][j] == color:
                # found a piece of the right color to flip the pieces between
                return (pieces_flipped, pieces)
        return 0, []

    def count_pieces(self):
        self.white_pieces = 0
        self.black_pieces = 0
        self.empty_spaces = 64
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == WHITE:
                    self.white_pieces += 1
                    self.empty_spaces -= 1
                elif self.board[i][j] == BLACK:
                    self.black_pieces += 1
                    self.empty_spaces -= 1

    def game_won(self):
        # Game Won if One Player Has No Pieces on the Board
        if self.white_pieces == 0:
            return BLACK
        elif self.black_pieces == 0:
            return WHITE
        # Game also over if no valid moves for both players or no empty spaces left on board
        elif (self.get_valid_moves(BLACK) == [] and self.get_valid_moves(WHITE) == []) or self.empty_spaces == 0:
            if self.white_pieces > self.black_pieces:
                return WHITE
            elif self.black_pieces > self.white_pieces:
                return BLACK
            else:
                return EMPTY # returning EMPTY denotes a tie
        return None

    """
    def child_nodes(self, color):
        moves = self.get_valid_moves(color)
        children = [None]*len(moves)
        for (i, move) in enumerate(moves):
            child = Board()
            child.now_playing = self.now_playing
            child.board = np.copy(self.board)
            child.valid_moves = self.valid_moves
            child.white_pieces = self.white_pieces
            child.black_pieces = self.black_pieces
            child.empty_spaces = self.empty_spaces
            child.apply_move(move, color)
            children[i] = (child, move)
        return children
    """

    # Function to print board for text based game
    def print_board(self):
        print('  ', end=' ')
        for i in range(8):
            print(' ', i, end=' ')
        print()
        for i in range(8):
            print(i, ' |', end=' ')
            for j in range(8):
                if self.board[i][j] == BLACK:
                    print('B', end=' ')
                elif self.board[i][j] == WHITE:
                    print('W', end=' ')
                else:
                    print(' ', end=' ')
                print('|', end=' ')
            print()

    def __lt__(self, other):
        return random.randint(0,1) -0.5

    def get_representation(self, color):
        """ Return a board where the current player is always black """
        if color == BLACK:
            return self.board.copy()

        representation = []
        for row in self.board:
            new_row = []
            for field in row:
                if field == EMPTY:
                    new_row.append(EMPTY)
                elif field == BLACK:
                    new_row.append(WHITE)
                else:
                    new_row.append(BLACK)
            representation.append(new_row)

        return np.array(representation, dtype=np.float64)

    def get_legal_moves_map(self, color):

        legal_moves_map = np.zeros([8, 8])
        for move in self.get_valid_moves(color):
            legal_moves_map[move[0]][move[1]] = 1

        return legal_moves_map

    def copy(self):
        return Board(np.copy(self.board))

