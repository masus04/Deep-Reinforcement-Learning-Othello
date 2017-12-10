from datetime import datetime

import src.config as config
from src.othello import Othello
from src.plotter import Printer
from src.player import HeuristicPlayer, ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer
from src.valueFunction import ValueFunction, SimpleValueFunction, FCValueFunction, DecoupledValueFunction

import evaluation

EXPERIMENT_NAME = "|TDvsMC|"


def conditional_evaluation(players, evaluation_games):
    for player in players:
        if player.train:
            evaluation.evaluate(player=player, games=evaluation_games, silent=True)


def train(player1, player2, games, evaluation_period, experiment_name=EXPERIMENT_NAME, silent=False):
    simulation = Othello(player1, player2)

    start_time = datetime.now()

    conditional_evaluation([player1, player2], 4)

    for i in range(games//evaluation_period):
        simulation.run_simulations(episodes=evaluation_period, silent=True)

        conditional_evaluation([player1, player2], 20)

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
    player1 = TDPlayer(color=config.BLACK, strategy=DecoupledValueFunction)
    player2 = HeuristicPlayer(color=config.WHITE, strategy=ValueFunction)

    """ Continue training """
    # player1 = config.load_player("TDPlayer_Black_ValueFunction|TDvsMC|")
    # player2 = config.load_player("MCPlayer_White_ValueFunction|TDvsMC|")

    TOTAL_GAMES = 1000
    EVALUATION_PERIOD = 250

    """ Execution """
    print("Experiment name: %s" % EXPERIMENT_NAME)
    print("Training %s VS %s" % (player1.player_name, player2.player_name))
    train(player1, player2, TOTAL_GAMES, EVALUATION_PERIOD)
    print("Training completed")
