
import src.config as config
from src.board import Board
from src.player import MCTSPlayer, TDPlayer
from src.valueFunction import ValueFunction

board = Board()
player = MCTSPlayer(config.BLACK, TDPlayer, ValueFunction)
player.get_move(board)
print(player.get_move(board))
