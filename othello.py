#!/usr/bin/env python

import board
import player
from gui import Gui
from config import BLACK, WHITE, HEADLESS, TIMEOUT, get_color_from_player_number


class Othello:

    def __init__(self):
        pass

    def setup_game(self):
        self.gui = Gui()

        self.now_playing = player.ComputerPlayer(BLACK, TIMEOUT, self.gui)
        self.other_player = player.ComputerPlayer(WHITE, TIMEOUT, self.gui)

        self.board = board.Board()

    def run(self):
        self.setup_game()
        if not HEADLESS:
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


def main():
    game = Othello()
    print("winner: %s" % get_color_from_player_number(game.run()))

if __name__ == '__main__':
    main()
