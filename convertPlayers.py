import core.config as config
from core.plotter import Plotter
from datetime import datetime

EXPERIMENT_NAME = "|Continuous|"

player1 = config.load_player("TDPlayer_Black_ValueFunction" + EXPERIMENT_NAME)
player2 = config.load_player("TDPlayer_White_ValueFunction" + EXPERIMENT_NAME)

"""
for player in player1, player2:
    player.plotter.plot_results(comment=COMMENT)
    player.plotter.plot_scores(comment=COMMENT)
    player.save(comment=COMMENT)
"""

for player in player1, player2:

    player.plotter.num_episodes = sum([opponent[1] for opponent in player.opponents])
    player.plotter = Plotter(player.plotter.plot_name, player.plotter)

    player.plotter.plot_results(experiment_name=EXPERIMENT_NAME)
    player.plotter.plot_scores(experiment_name=EXPERIMENT_NAME)
    player.save(comment=EXPERIMENT_NAME)
