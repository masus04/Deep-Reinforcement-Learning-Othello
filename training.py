import sys
import src.config as config
from src.othello import Othello
from src.player import HeuristicPlayer, ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer
from src.valueFunction import SimpleValueFunction, FCValueFunction

player1 = TDPlayer(color=config.BLACK, strategy=SimpleValueFunction)
player2 = TDPlayer(color=config.WHITE, strategy=SimpleValueFunction)

simulation = Othello(player1, player2)

""" Continue training """
# player1.load_params()
# player2.load_params()

""" | Training script | """

TOTAL_GAMES = 100000

# training
print("\nStarted training")
simulation.run_simulations(TOTAL_GAMES)

# save artifacts
for player in (player1, player2):
    player.plotter.plot_results(resolution=100)
    player.save()

""" | Training script | """

print("Training completed")
