#!/usr/bin/env python3
import src.board as board
from src.gui import Gui, NoGui
from src.config import HEADLESS, get_color_from_player_number


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
                self.now_playing.register_winner(winner)
                self.other_player.register_winner(winner)
                return winner
            valid_moves = self.board.get_valid_moves(self.now_playing.color)
            self.gui.highlight_valid_moves(valid_moves)
            if valid_moves != []:
                move = self.now_playing.get_move(self.board)
                self.gui.flash_move(move, self.now_playing.color)
                if not move in valid_moves:
                    raise Exception("Player %s performed an illegal move: %s" % (get_color_from_player_number(self.now_playing.color), move))
                self.board.apply_move(move, self.now_playing.color)
            self.gui.update(self.board, self.other_player)
            self.now_playing, self.other_player = self.other_player, self.now_playing
