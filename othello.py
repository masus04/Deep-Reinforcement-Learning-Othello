#!/usr/bin/env python

import board
import player
from gui import Gui, NoGui
from config import BLACK, WHITE, HEADLESS, TIMEOUT, get_color_from_player_number
from heuristic import OthelloHeuristic


class Othello:

    def __init__(self, player1, player2):
        self.gui = NoGui() if HEADLESS else Gui()

        self.now_playing = player1.set_gui(self.gui)
        self.other_player = player2.set_gui(self.gui)

    def run(self):
        self.board = board.Board()
        self.gui.show_game(self.board)
        while True:
            winner = self.board.game_won()
            if winner is not None:
                return winner
            valid_moves = self.board.get_valid_moves(self.now_playing.color)
            if valid_moves != []:
                move = self.now_playing.get_move(self.board)
                self.gui.flash_move(move, self.now_playing.color)
                if not move in valid_moves:
                    raise Exception("Player %s performed an illegal move: %s" % (get_color_from_player_number(self.now_playing.color), move))
                self.board.apply_move(move, self.now_playing.color)
            self.gui.update(self.board, self.other_player)
            self.now_playing, self.other_player = self.other_player, self.now_playing

    def print_winner(self):
        print(("Winner: %s" % get_color_from_player_number(self.run())))
