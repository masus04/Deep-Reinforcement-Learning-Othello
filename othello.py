#!/usr/bin/env python

import board
import player
from gui import Gui, NoGui
from config import BLACK, WHITE, HEADLESS, TIMEOUT, get_color_from_player_number
from heuristic import OthelloHeuristic


class Othello:

    def __init__(self, player1, player2):
        self.gui = NoGui() if HEADLESS else Gui()

        self.now_playing = player1(BLACK, TIMEOUT, self.gui, OthelloHeuristic.DEFAULT_STRATEGY)
        self.other_player = player2(WHITE, TIMEOUT, self.gui, OthelloHeuristic.DEFAULT_STRATEGY)

    def run(self):
        self.board = board.Board()
        self.gui.show_game(self.board)
        while True:
            winner = self.board.game_won()
            if winner is not None:
                return winner

            self.now_playing.set_current_board(self.board)
            if self.board.get_valid_moves(self.now_playing.color) != []:
                self.board = self.now_playing.get_move()
            self.gui.update(self.board, self.other_player)
            self.now_playing, self.other_player = self.other_player, self.now_playing

    def print_winner(self):
        print(("Winner: %s" % get_color_from_player_number(self.run())))
