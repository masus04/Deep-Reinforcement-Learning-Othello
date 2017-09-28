import sys
import src.config as config
from src.othello import Othello
from src.player import ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer
from src.plotter import print_inplace
from datetime import datetime

player1 = TDPlayer(color=config.BLACK)
player2 = TDPlayer(color=config.WHITE)

simulation = Othello(player1, player2)

""" Continue training """
# player1.load_params()
# player2.load_params()

""" | Training script | """

TOTAL_GAMES = 100000
EVALUATION_GAMES = 0

# training
print("Started training")
simulation.run_training_simulations(TOTAL_GAMES-EVALUATION_GAMES, cuda=True, silent="-silent" in sys.argv or "-s" in sys.argv)

# evaluation
print("Started evaluation")

""" | Training script | """
