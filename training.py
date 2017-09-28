import sys
import src.config as config
from src.othello import Othello
from src.player import ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer

silent = "-silent" in sys.argv or "-s" in sys.argv
print("\nSilent mode: %s" % silent)

player1 = TDPlayer(color=config.BLACK)
player2 = RandomPlayer(color=config.WHITE)

simulation = Othello(player1, player2)

""" Continue training """
# player1.load_params()
# player2.load_params()

""" | Training script | """

TOTAL_GAMES = 100
EVALUATION_GAMES = 0

# training
print("Started training")
simulation.run_training_simulations(TOTAL_GAMES-EVALUATION_GAMES, cuda=True, silent=silent)

# save artifacts
for player in (player1, player2):
    player.plotter.plot_results(resolution=200)
    player.save_params()

# evaluation
print("Started evaluation")

""" | Training script | """

print("Training completed")
