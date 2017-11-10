
import src.config as config
from src.player import MCTSPlayer, TDPlayer
from src.valueFunction import ValueFunction
from evaluation import compare_players

""" Play td_black against itself, once with and once without MCTS """

td_white = config.load_player("TDPlayer_Black_ValueFunction|TD vs MC|")
td_white.color = config.WHITE

for i in [2, 5, 10, 20, 25, 50, 100, 150, 300, 1000]:
    td_black = MCTSPlayer(config.BLACK, TDPlayer, ValueFunction, tree_exploration_constant=1/i)

    for player in td_black, td_white:
        player.explore = False

    print()
    compare_players(player1=td_black, player2=td_white, games=10)
