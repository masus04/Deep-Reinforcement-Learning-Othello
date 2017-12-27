import core.config as config
from core.board import Board
from core.player import MCPlayer
from core.valueFunction import ValueFunction

board = Board()
player = MCPlayer(color=config.BLACK, e=config.EPSILON, strategy=ValueFunction(), time_limit=config.TIMEOUT)

print(player.get_move(board))
player.save()
