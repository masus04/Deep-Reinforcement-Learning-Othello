import src.config as config
from src.board import Board
from src.player import MCPlayer
from src.valueFunction import ValueFunction

board = Board()
player = MCPlayer(color=config.BLACK, e=config.EPSILON, strategy=ValueFunction(), time_limit=config.TIMEOUT)

print(player.get_move(board))
player.save()
