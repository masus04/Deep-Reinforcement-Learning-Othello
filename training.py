from datetime import datetime

import src.config as config
from src.othello import Othello
from src.plotter import Printer
from src.player import HeuristicPlayer, ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer
from src.valueFunction import ValueFunction, SimpleValueFunction, FCValueFunction

import evaluation

EXPERIMENT_NAME = "|TDvsHeuristic|"


def train(player1, player2, games, evaluation_period, experiment_name=EXPERIMENT_NAME, silent=False):
    simulation = Othello(player1, player2)

    """ Actual training """
    start_time = datetime.now()
    evaluation.evaluate(player=player1, games=4, silent=True)
    evaluation.evaluate(player=player2, games=4, silent=True)
    for i in range(games//evaluation_period):
        # Training
        simulation.run_simulations(episodes=evaluation_period, silent=True)
        # Evaluation
        evaluation.evaluate(player=player1, games=20, silent=True)
        evaluation.evaluate(player=player2, games=20, silent=True)
        if not silent:
            Printer.print_inplace("Episode %s/%s" % (evaluation_period*(i+1), games), evaluation_period*(i + 1) / games * 100, datetime.now() - start_time)

        # save artifacts
        player1.plotter.clear_plots(experiment_name)
        for player in (player1, player2):
            player.plotter.plot_results(experiment_name)
            player.plotter.plot_scores(experiment_name)
            player.save(experiment_name)


if __name__ == "__main__":

    """ Parameters """
    player1 = TDPlayer(color=config.BLACK, strategy=ValueFunction)
    player2 = HeuristicPlayer(color=config.WHITE, strategy=ValueFunction)

    """ Continue training """
    # player1.load_player(color=config.BLACK, strategy=ValueFunction)
    # player2.load_player(color=config.WHITE, strategy=ValueFunction)

    TOTAL_GAMES = 250000
    EVALUATION_PERIOD = 1000

    """ Execution """
    print("Experiment name: %s" % EXPERIMENT_NAME)
    print("Training %s VS %s" % (player1.player_name, player2.player_name))
    train(player1, player2, TOTAL_GAMES, EVALUATION_PERIOD)
    print("Training completed")
