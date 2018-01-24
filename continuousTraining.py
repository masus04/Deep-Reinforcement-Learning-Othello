from datetime import datetime

import core.config as config
from training import train, generate_and_save_artefacts
from evaluation import compare_players, evaluate_all
from core.player import MCPlayer, TDPlayer, ReinforcePlayer
from core.valueFunction import ValueFunction, SimpleValueFunction, HugeDecoupledValueFunction, HugeValueFunction, LargeDecoupledValueFunction, LargeValueFunction
from core.plotter import Printer


def train_continuous(player1, player2, games, experiment_name, iterations, start_time=datetime.now()):
    """Trains a pair of players for @games, selects the stronger of both to continue and repeats the process for @iterations"""
    print("Experiment name: %s" % experiment_name)

    # Initial evaluation
    evaluate_all([player1, player2], 8)

    for i in range(iterations):
        train(player1, player2, games, experiment_name)
        evaluate_all([player1, player2], 20)
        generate_and_save_artefacts([player1, player2], experiment_name)

        player1, player2 = (player1, player2) if compare_players(player1, player2, silent=(i != iterations-1)) >= 0 else (player2, player1)
        player2 = player1.copy_with_inversed_color()

        print("Iteration %s/%s Simulation time: %s\n" % (i, iterations, str(datetime.now()-start_time).split(".")[0]))

    return player1, player2


def train_continuous_asymmetrical(player1, games, experiment_name, iterations, start_time=datetime.now(), best=None):
    """"Only train player1 while player2 is fixed to the currently best iteration and does not train"""
    print("Experiment name: %s" % experiment_name)

    if not best:
        best = player1.copy_with_inversed_color()
        best.add_to_name("-BEST-")
        best.replaced = []

    # Initial evaluation
    evaluate_all([player1, best], 8)

    # continuously improve
    for i in range(iterations):
        best.train = False

        train(player1, best, games, silent=True)
        evaluate_all([player1, best], 16)
        generate_and_save_artefacts([player1, best], experiment_name)

        if compare_players(player1, best, games=40, silent=(i != iterations-1)) >= 0:
            best.value_function = player1.value_function.copy()
            best.plotter = player1.plotter.copy()
            best.opponents = player1.opponents.copy()
            best.replaced.append(i)

        Printer.print_inplace(text="Iteration %s/%s" % (i+1, iterations), percentage=100 * (i+1) / (iterations), time_taken=str(datetime.now() - start_time).split(".")[0],
                              comment=" | Best player replaced at: %s\n" % best.replaced)

    print()
    evaluate_all([player1, best], 80, silent=False)
    return player1, best


if __name__ == "__main__":

    """ Parameters """
    PLAYER = ReinforcePlayer(color=config.BLACK, lr=0.1, alpha=0.003, e=0.001)
    PLAYER2 = None

    # PLAYER = config.load_player("TDPlayer_Black_ValueFunction|Async|")
    # PLAYER2 = config.load_player("TDPlayer_White_ValueFunction_BEST|Async|")

    # PLAYER = config.load_player("TDPlayer_Black_ValueFunction|Continuous|")
    # PLAYER2 = config.load_player("TDPlayer_White_ValueFunction|Continuous|")

    assert PLAYER.color == config.BLACK
    # assert PLAYER2.color == config.WHITE

    ITERATIONS = 100
    GAMES_PER_ITERATION = 5000

    """ Execution """
    start = datetime.now()
    # train_continuous(player1=PLAYER, player2=PLAYER2, games=GAMES_PER_ITERATION, experiment_name="|Continuous|training lr:%s a:%s|" % (PLAYER.value_function.learning_rate, PLAYER.alpha), iterations=ITERATIONS)
    player1, best = train_continuous_asymmetrical(player1=PLAYER, best=PLAYER2, games=GAMES_PER_ITERATION, experiment_name="|Async training|", iterations=ITERATIONS)

    print("Training completed, took %s" % str(datetime.now()-start).split(".")[0])
