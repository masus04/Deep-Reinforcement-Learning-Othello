#!/usr/bin/env python3
from datetime import datetime

import src.config as config
from src.board import Board
from src.gui import Gui, NoGui
from src.config import HEADLESS, get_color_from_player_number
from src.plotter import Printer


class Othello:

    def __init__(self, player1, player2, headless=HEADLESS):
        self.gui = NoGui() if headless else Gui()

        self.player1 = player1.set_gui(self.gui)
        self.player2 = player2.set_gui(self.gui)
        self.printer = Printer()

    def run(self, player1, player2):
        self.board = Board()

        self.now_playing = player1
        self.other_player = player2

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
                    raise Exception("Player %s(%s) performed an illegal move: %s" % (self.now_playing.player_name, get_color_from_player_number(self.now_playing.color), move))
                self.board.apply_move(move, self.now_playing.color)
            self.gui.update(self.board, self.other_player)
            self.now_playing, self.other_player = self.other_player, self.now_playing

    def run_simulations(self, episodes, silent=False):
        colors = (self.player1.color, self.player2.color)

        self.player1.add_opponent(self.player2)
        self.player2.add_opponent(self.player1)
        players = [self.player1, self.player2]
        results = []

        start_time = datetime.now()
        for i in range(episodes):
            winner = self.run(players[i % 2], players[(i + 1) % 2])
            result = config.get_result_label(winner == self.player1.color)
            results.append(result)
            if self.player1.train:
                players[0].plotter.add_result(result)
                players[1].plotter.add_result((result+1) % 2)

            if not silent:
                self.printer.print_inplace("Episode %s/%s" % (i + 1, episodes), (i + 1) / episodes * 100, datetime.now() - start_time)

            if (i+1) % 2 == 0:  # switch colors every 2 rounds
                self.player1.color, self.player2.color = self.player2.color, self.player1.color

        self.player1.color, self.player2.color = colors

        return results
