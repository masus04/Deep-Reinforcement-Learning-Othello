from datetime import datetime
from math import sqrt

import src.config as config
from src.othello import Othello
from src.plotter import Printer
from src.player import HeuristicPlayer, ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer
from src.valueFunction import ValueFunction, SimpleValueFunction, FCValueFunction
import evaluation


def train(player1, player2, games, evaluation_period):
    simulation = Othello(player1, player2)
    printer = Printer()

    """ Continue training """
    # player1.load_player(color=config.BLACK, strategy=ValueFunction)
    # player2.load_player(color=config.WHITE, strategy=ValueFunction)

    # player1.value_function = player2.value_function.copy(player1.plotter)

    """ Actual training """
    start_time = datetime.now()
    print("Training %s & %s" % (player1.player_name, player2.player_name))
    evaluation.evaluate(player=player1, games=4, silent=True)
    evaluation.evaluate(player=player2, games=4, silent=True)
    for i in range(games//evaluation_period):
        # Training
        simulation.run_simulations(episodes=evaluation_period, clear_plots=True, silent=True)
        # Evaluation
        evaluation.evaluate(player=player1, games=int(sqrt(games/10)), silent=True)
        evaluation.evaluate(player=player2, games=int(sqrt(games/10)), silent=True)
        printer.print_inplace("Episode %s/%s" % (evaluation_period*(i+1), games), evaluation_period*(i + 1) / games * 100, datetime.now() - start_time)


if __name__ == "__main__":

    player1 = TDPlayer(color=config.BLACK, strategy=ValueFunction, lr=0.1, alpha=0.01)
    player2 = TDPlayer(color=config.WHITE, strategy=ValueFunction)

    TOTAL_GAMES = 50000
    EVALUATION_PERIOD = 100

    train(player1, player2, TOTAL_GAMES, EVALUATION_PERIOD)

    # save artifacts
    for player in (player1, player2):
        player.plotter.plot_results()
        player.plotter.plot_scores()
        player.save()

    print("Training completed")
