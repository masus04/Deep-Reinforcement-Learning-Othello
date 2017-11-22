import src.config as config
from src.plotter import Plotter
from datetime import datetime

COMMENT = "|Async|"

player1 = config.load_player("TDPlayer_Black_ValueFunction" + COMMENT)
player2 = config.load_player("TDPlayer_White_ValueFunction_BEST" + COMMENT)

for player in player1, player2:

    player.plotter.num_episodes = sum([opponent[1] for opponent in player.opponents])
    player.plotter = Plotter(player.plotter.plot_name, player.plotter)
    player.value_function.plotter = player.plotter

    player.plotter.plot_results(comment=COMMENT)
    player.plotter.plot_scores(comment=COMMENT)
    player.save(comment=COMMENT)
