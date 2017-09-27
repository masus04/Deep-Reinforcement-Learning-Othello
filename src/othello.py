#!/usr/bin/env python3
from datetime import datetime

import src.board as board
from src.gui import Gui, NoGui
from src.config import HEADLESS, get_color_from_player_number
from src.plotter import print_inplace


class Othello:

    def __init__(self, player1, player2):
        self.gui = NoGui() if HEADLESS else Gui()

        self.players1 = player1.set_gui(self.gui)
        self.players2 = player2.set_gui(self.gui)

    def run(self, player1, player2):
        self.board = board.Board()

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
                    raise Exception("Player %s performed an illegal move: %s" % (get_color_from_player_number(self.now_playing.color), move))
                self.board.apply_move(move, self.now_playing.color)
            self.gui.update(self.board, self.other_player)
            self.now_playing, self.other_player = self.other_player, self.now_playing

    def run_training_simulations(self, episodes, cuda=True):
        players = [self.players1, self.players2]
        for player in players:
            player.value_function.cuda = cuda

        start_time = datetime.now()
        for i in range(episodes):
            result = self.run(players[i % 2], players[(i+1) % 2])
            for player in players:
                player.plotter.add_result(result)

            print_inplace("Episode %s/%s" % (i + 1, episodes), (i + 1) / episodes * 100, datetime.now() - start_time)
        print()
