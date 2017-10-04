import sys
import src.config as config
from src.othello import Othello
from src.player import ComputerPlayer, RandomPlayer, HeuristicPlayer, MCPlayer, TDPlayer
from src.valueFunction import ValueFunction, SimpleValueFunction

td_black = TDPlayer.load_player(color=config.BLACK, strategy=SimpleValueFunction)
mc_player = MCPlayer.load_player(color=config.BLACK, strategy=ValueFunction)
heuristic_black = HeuristicPlayer(color=config.BLACK)
random_black = RandomPlayer(color=config.BLACK)

td_white = TDPlayer.load_player(color=config.WHITE, strategy=SimpleValueFunction)
heuristic_white = HeuristicPlayer(color=config.WHITE)
random_white = RandomPlayer(color=config.WHITE)

players = [td_black, td_white]
reference_players = [[td_white, heuristic_white, random_white], [td_black, heuristic_black, random_black]]

EVALUATION_GAMES = 10

print("Evaluation:")
for i, player in enumerate(players):
    player.train = False

    for reference_player in reference_players[i]:
        reference_player.train = False
        simulation = Othello(player, reference_player)
        results = simulation.run_simulations(EVALUATION_GAMES)

        print("%s won %s of games against %s\n" % (player.player_name, "{0:.3g}".format((sum(results)/EVALUATION_GAMES) * 100) + "%", reference_player.player_name))
