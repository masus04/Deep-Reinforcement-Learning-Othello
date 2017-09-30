import sys
import src.config as config
from src.othello import Othello
from src.player import ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer

player1 = TDPlayer(color=config.BLACK)
player2 = TDPlayer(color=config.WHITE)
player2.color = config.BLACK  # load both black and white during training but use them as black for evaluation

reference_players = [RandomPlayer(color=config.WHITE)]

players = [player1, player2]

EVALUATION_GAMES = 100

print("Evaluation:")
for player in players:
    player.load_params()
    player.train = False

    for reference_player in reference_players:
        simulation = Othello(player, reference_player)
        results = simulation.run_simulations(EVALUATION_GAMES)

        print("%s won %s of games agains %s" % (player.player_name, "{0:.3g}".format((sum(results)/EVALUATION_GAMES-config.BLACK) * 100) + "%", reference_player.player_name))
