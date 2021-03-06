from datetime import datetime

import core.config as config
from core.othello import Othello
from core.plotter import Printer
from core.player import HeuristicPlayer, ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer
import core.valueFunction as vF

import evaluation

EXPERIMENT_NAME = "|Training|"


def train(player1, player2, games, experiment_name=EXPERIMENT_NAME, silent=False):
    simulation = Othello(player1, player2)
    simulation.run_simulations(episodes=games, silent=silent)


def generate_and_save_artefacts(players, experiment_name):
    for player in players:
        player.plotter.clear_plots(experiment_name)
        player.plotter.plot_results(experiment_name)
        player.plotter.plot_scores(experiment_name)
        player.save(experiment_name)
        # player.save("%s lr:%s" % (experiment_name, player.value_function.learning_rate))


def train_and_evaluate(player1, player2, games, evaluation_period, experiment_name=EXPERIMENT_NAME, silent=False):
    simulation = Othello(player1, player2)

    start_time = datetime.now()

    for i in range(games//evaluation_period):
        simulation.run_simulations(episodes=evaluation_period, silent=silent)
        evaluation.evaluate_all([player1, player2], 20)

        if not silent:
            Printer.print_inplace("Episode %s/%s" % (evaluation_period*(i+1), games), evaluation_period*(i + 1) / games * 100, datetime.now() - start_time)

        # save artifacts
        player1.plotter.clear_plots(experiment_name)
        for player in (player1, player2):
            player.plotter.plot_results(experiment_name)
            player.plotter.plot_scores(experiment_name)
            player.save(experiment_name)


if __name__ == "__main__":
    """ This script is run in order to test if all available ValueFunctions can be trained as expected """

    strategies = [vF.LargeValueFunction]

    for strategy in strategies:
        """ Parameters """
        player1 = TDPlayer(color=config.BLACK, strategy=strategy, lr=0.1)
        player2 = HeuristicPlayer(color=config.WHITE, strategy=vF.NoValueFunction)

        """ Continue training """
        # player1 = config.load_player("TDPlayer_Black_ValueFunction|TDvsMC|")
        # player2 = config.load_player("MCPlayer_White_ValueFunction|TDvsMC|")

        TOTAL_GAMES = 200000
        EVALUATION_PERIOD = TOTAL_GAMES//4

        """ Execution """
        start = datetime.now()
        print("Experiment name: %s" % EXPERIMENT_NAME)
        print("Training %s VS %s" % (player1.player_name, player2.player_name))
        evaluation.evaluate_all([player1, player2], 8)
        train_and_evaluate(player1, player2, TOTAL_GAMES, EVALUATION_PERIOD)
        print("Training completed, took %s\n" % (datetime.now() - start))
