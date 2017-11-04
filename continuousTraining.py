from datetime import datetime

import src.config as config
from training import train
from evaluation import compare_players
from src.player import MCPlayer, TDPlayer
from src.valueFunction import ValueFunction

EXPERIMENT_NAME = "|continuous|"


def train_continuous(player1, player2, iterations):

    for i in range(iterations):
        train(player1, player2, GAMES_PER_ITERATION, EVALUATION_PERIOD, EXPERIMENT_NAME)
        player1, player2 = (player1, player2) if compare_players(player1, player2, silent=(i == iterations-1)) >= 0 else (player2, player1)
        player2.value_function = player1.value_function.copy(player2.plotter)


if __name__ == "__main__":

    """ Parameters """
    PLAYER = TDPlayer
    ITERATIONS = 2
    GAMES_PER_ITERATION = 50
    EVALUATION_PERIOD = 25

    """ Execution """
    print("Experiment name: %s" % EXPERIMENT_NAME)
    train_continuous(PLAYER(config.BLACK, ValueFunction), PLAYER(config.WHITE, ValueFunction), ITERATIONS)
    print("Training completed")
