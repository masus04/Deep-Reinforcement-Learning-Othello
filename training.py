from datetime import datetime

import core.config as config
from core.othello import Othello
from core.plotter import Printer
from core.player import HeuristicPlayer, ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer, ReinforcePlayer
import core.valueFunction as vF

import evaluation

EXPERIMENT_NAME = "|Training|"


def train(player1, player2, games, experiment_name=EXPERIMENT_NAME, silent=False):
    simulation = Othello(player1, player2)
    return simulation.run_simulations(episodes=games, silent=silent)


def generate_and_save_artefacts(players, experiment_name):
    for player in players:
        player.plotter.clear_plots(experiment_name)
        player.plotter.plot_results(experiment_name)
        player.plotter.plot_scores(experiment_name)
        player.save(experiment_name)
        # player.save("%s lr:%s" % (experiment_name, player.value_function.learning_rate))


def train_and_evaluate(player1, player2, games, evaluation_period, experiment_name=EXPERIMENT_NAME, silent=False, plot_only_p1=False):
    simulation = Othello(player1, player2)

    start_time = datetime.now()

    for i in range(games//evaluation_period):
        simulation.run_simulations(episodes=evaluation_period, silent=silent)
        evaluation.evaluate_all([player1, player2], 20)

        if not silent:
            Printer.print_inplace("Episode %s/%s" % (evaluation_period*(i+1), games), evaluation_period*(i + 1) / games * 100, datetime.now() - start_time)

        # save artifacts
        for player in [player1] if plot_only_p1 else [player1, player2]:
            player.plotter.clear_plots(experiment_name)
            player.plotter.plot_results(experiment_name)
            player.plotter.plot_scores(experiment_name)
            player.save(experiment_name)


if __name__ == "__main__":
    """ This script is run in order to test if all available ValueFunctions can be trained as expected """

    strategies = [vF.LargeValueFunction]

    for strategy in strategies:
        """ Parameters """
        # player1 = ReinforcePlayer(color=config.BLACK, lr=0.001)
        # player2 = ReinforcePlayer(color=config.WHITE, lr=0.0001)

        """ Continue training """
        player1 = config.load_player("ReinforcePlayer_Black_PGValueFunction|Training|")
        player2 = config.load_player("ReinforcePlayer_White_PGValueFunction|Training|")

        assert player1.color == config.BLACK
        assert player2.color == config.WHITE

        TOTAL_GAMES = 1000000
        EVALUATION_PERIOD = TOTAL_GAMES//100

        """ Execution """
        start = datetime.now()
        print("Experiment name: %s" % EXPERIMENT_NAME)
        print("Training %s VS %s" % (player1.player_name, player2.player_name))
        evaluation.evaluate_all([player1, player2], 8)
        train_and_evaluate(player1, player2, TOTAL_GAMES, EVALUATION_PERIOD)
        print("Training completed, took %s\n" % (datetime.now() - start))
