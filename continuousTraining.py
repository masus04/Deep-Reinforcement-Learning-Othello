from datetime import datetime

import src.config as config
from training import train
from evaluation import compare_players
from src.player import MCPlayer, TDPlayer
from src.valueFunction import ValueFunction

EXPERIMENT_NAME = "|Continuous Training|"


def train_continuous(player1, player2, games, evaluation_period, experiment_name, iterations):
    """Trains a pair of players for @games games, evaluating them every @evaluation_period games and repeats the process with the stronger of the two players for @iterations iterations"""

    for i in range(iterations):
        train(player1, player2, games, evaluation_period, experiment_name)
        player1, player2 = (player1, player2) if compare_players(player1, player2, silent=(i != iterations-1)) >= 0 else (player2, player1)
        player2.value_function = player1.value_function.copy()
        print("Simulation time: %s\n" % str(datetime.now()-start).split(".")[0])

    return player1, player2


if __name__ == "__main__":

    """ Parameters """
    PLAYER = TDPlayer
    ITERATIONS = 10
    GAMES_PER_ITERATION = 50000
    EVALUATION_PERIOD = 5000

    """ Execution """
    start = datetime.now()
    print("Experiment name: %s" % EXPERIMENT_NAME)
    train_continuous(player1=PLAYER(config.BLACK, ValueFunction), player2=PLAYER(config.WHITE, ValueFunction), games=GAMES_PER_ITERATION, evaluation_period=EVALUATION_PERIOD, experiment_name=EXPERIMENT_NAME, iterations=ITERATIONS)
    print("Training completed, took %s" % str(datetime.now()-start).split(".")[0])
