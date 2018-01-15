from datetime import datetime

import core.config as config
from core.othello import Othello
from core.player import HeuristicPlayer, ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer
from core.valueFunction import ValueFunction, SimpleValueFunction, FCValueFunction
import training
from evaluation import evaluate

PLAYER = TDPlayer
EXPERIMENT_NAME = "|CROSS_VALIDATION|"

evaluation_file = open("./plots/crossEvaluation_%s.txt" % PLAYER.__name__, "w+")


def log_message(message):
    print(message)
    evaluation_file.write(message + "\n")


def evaluation(games, evaluation_games, lr, e, a, experiment_name):
    log_message("Evaluating lr:%s alpha:%s epsilon:%s " % (lr, a, e))

    player1 = PLAYER(color=config.BLACK, strategy=ValueFunction, lr=lr, e=e, alpha=a)
    player2 = PLAYER(color=config.WHITE, strategy=ValueFunction, lr=lr, e=e, alpha=a)
    players = [player1, player2]

    """ Training """

    training.train(player1=player1, player2=player2, games=games)

    """ Evaluation """

    player1.train = False
    last_10 = player1.plotter.last10Results.get_values()
    player1.score = sum(last_10)/len(last_10)
    # player1.plotter.evaluation_scores = player1.plotter.evaluation_scores.get_values()[:-1]  # Replace last evaluation with a more accurate one
    # evaluate(player=player1, games=evaluation_games, log_method=log_message, silent=True)
    player1.plotter.plot_results(experiment_name=experiment_name)
    # player1.plotter.plot_scores(comment=" e:%s, a:%s" % (e, a))

    log_message("|--- LR:%s Alpha:%s Epsilon:%s Score:%s Simulation time: %s ---|\n" % (lr, a, e, player1.score, str(datetime.now() - start_time).split(".")[0]))

    return player1.score


if __name__ == "__main__":

    start_time = datetime.now()

    lr = 0.1
    epsilons = [0.001, 0.0003, 0.0001, 0.00003, 0.00001]
    alphas = [0.03, 0.01, 0.003, 0.001]

    TRAINING_GAMES = 200000     # Total training games per configuration
    EVALUATION_GAMES = 100      # Number of final evaluation games

    results = [(evaluation(TRAINING_GAMES, EVALUATION_GAMES, lr, e, a, EXPERIMENT_NAME), lr, e, a) for e in epsilons for a in alphas]
    for r in sorted(results, reverse=True):
        log_message("score:%s lr:%s e:%s a:%s" % r)

    print("\nCrossValidation timer: %s" % str(datetime.now() - start_time).split(".")[0])
    evaluation_file.close()
