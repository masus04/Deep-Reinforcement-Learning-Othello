import sys
import src.config as config
from src.othello import Othello
from src.player import ComputerPlayer, RandomPlayer, HeuristicPlayer, MCPlayer, TDPlayer

td_black = TDPlayer(color=config.BLACK)
mc_player = MCPlayer(color=config.BLACK)
heuristic_black = HeuristicPlayer(color=config.BLACK)
random_black = RandomPlayer(color=config.BLACK)

td_white = TDPlayer(color=config.WHITE)
heuristic_white = HeuristicPlayer(color=config.WHITE)
random_white = RandomPlayer(color=config.WHITE)

players = [td_black]
reference_players = [td_white, heuristic_white, random_white]

EVALUATION_GAMES = 5000

print("Evaluation:")
for player in players:
    player.load_params()
    player.train = False

    for reference_player in reference_players:
        reference_player.load_params()
        reference_player.train = False
        simulation = Othello(player, reference_player)
        results = simulation.run_simulations(EVALUATION_GAMES)

        print("%s won %s of games against %s\n" % (player.player_name, "{0:.3g}".format((sum(results)/EVALUATION_GAMES) * 100) + "%", reference_player.player_name))

"""
player = td_player1
reference_player = heuristic_player
simulation = Othello(player, reference_player)
results = simulation.run_simulations(EVALUATION_GAMES)
print("%s won %s of games against %s\n" % (player.player_name, "{0:.3g}".format((sum(results)/EVALUATION_GAMES) * 100) + "%", reference_player.player_name))
"""
