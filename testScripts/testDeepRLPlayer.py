import config
from board import Board
from player import MCPlayer
from valueFunction import ValueFunction

board = Board()
player = MCPlayer(color=config.BLACK, time_limit=config.TIMEOUT, strategy=ValueFunction())

print(player.get_move(board))
player.save_params()
