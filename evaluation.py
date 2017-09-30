import sys
import src.config as config
from src.othello import Othello
from src.player import ComputerPlayer, RandomPlayer, HeuristicPlayer, MCPlayer, TDPlayer

td_player1 = TDPlayer(color=config.BLACK)
mc_player = MCPlayer(color=config.BLACK)

td_player2 = TDPlayer(color=config.WHITE)
heuristic_player = HeuristicPlayer(color=config.WHITE)
random_player = RandomPlayer(color=config.WHITE)

players = [td_player1, mc_player, RandomPlayer(color=config.BLACK)]
reference_players = [td_player2, random_player, heuristic_player]

EVALUATION_GAMES = 100

print("Evaluation:")
for player in players:
    player.load_params()
    player.train = False

    for reference_player in reference_players:
        reference_player.train = False
        simulation = Othello(player, reference_player)
        results = simulation.run_simulations(EVALUATION_GAMES)

        print("%s won %s of games against %s\n" % (player.player_name, "{0:.3g}".format((sum(results)/EVALUATION_GAMES) * 100) + "%", reference_player.player_name))
