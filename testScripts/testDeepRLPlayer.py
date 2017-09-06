import config
from board import Board
from player import DeepRLPlayer
from valueFunction import ValueFunction

board = Board()
player = DeepRLPlayer(color=config.BLACK, time_limit=config.TIMEOUT, strategy=ValueFunction())

print(player.get_move(board))
