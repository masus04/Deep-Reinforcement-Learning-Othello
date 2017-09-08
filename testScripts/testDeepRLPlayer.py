import config
from board import Board
from player import MCPlayer
from valueFunction import ValueFunction

board = Board()
player = MCPlayer(color=config.BLACK, e=config.EPSILON, strategy=ValueFunction(), time_limit=config.TIMEOUT)

print(player.get_move(board))
player.save_params()
