import sys
import src.config as config
from src.othello import Othello
from src.player import HeuristicPlayer, ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer

player1 = TDPlayer(color=config.BLACK)
player2 = HeuristicPlayer(color=config.WHITE)

simulation = Othello(player1, player2)

""" Continue training """
# player1.load_params()
# player2.load_params()

""" | Training script | """

TOTAL_GAMES = 100000
EVALUATION_GAMES = 0

# training
print("\nStarted training")
simulation.run_simulations(TOTAL_GAMES-EVALUATION_GAMES)

# save artifacts
for player in (player1, player2):
    player.plotter.plot_results(resolution=100)
    player.save_params()

""" | Training script | """

print("Training completed")
