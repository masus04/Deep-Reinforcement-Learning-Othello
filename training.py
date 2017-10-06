import src.config as config
from src.othello import Othello
from src.player import HeuristicPlayer, ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer
from src.valueFunction import ValueFunction, SimpleValueFunction, FCValueFunction

player1 = TDPlayer(color=config.BLACK, strategy=ValueFunction, lr=0.1, alpha=0.01)
player2 = TDPlayer(color=config.WHITE, strategy=ValueFunction)

simulation = Othello(player1, player2)

""" Continue training """
# player1.load_player(color=config.BLACK, strategy=ValueFunction)
# player2.load_player(color=config.WHITE, strategy=ValueFunction)

# player1.value_function = player2.value_function.copy(player1.plotter)

""" | Training script | """

TOTAL_GAMES = 100000

# training
print("\nStarted training")
simulation.run_simulations(TOTAL_GAMES, clear_plots=True)

# save artifacts
for player in (player1, player2):
    player.plotter.plot_results(resolution=100)
    player.save()

""" | Training script | """

print("Training completed")
