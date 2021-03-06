from datetime import datetime

import core.config as config
from core.othello import Othello
from core.player import HeuristicPlayer, ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer
from core.valueFunction import ValueFunction, SimpleValueFunction, FCValueFunction
import training
from evaluation import evaluate

PLAYER = TDPlayer
EXPERIMENT_NAME = "CROSS_VALIDATION"

evaluation_file = open("./plots/crossEvaluation_%s.txt" % PLAYER.__name__, "w+")


def log_message(message):
    print(message)
    evaluation_file.write(message + "\n")


def evaluation(games, evaluation_period, evaluation_games, lr, a):
    log_message("\nEvaluating LR:%s a:%s" % (lr, a))

    player1 = PLAYER(color=config.BLACK, strategy=ValueFunction, lr=lr, alpha=a)
    player2 = PLAYER(color=config.WHITE, strategy=ValueFunction, lr=lr, alpha=a)
    players = [player1, player2]

    """ Training """

    training.train_and_evaluate(player1=player1, player2=player2, games=games, evaluation_period=evaluation_period)

    """ Evaluation """

    for player in players:
        player.train = False
        player.score = 0
        player.plotter.evaluation_scores = player.plotter.evaluation_scores.get_values()[:-1]  # Replace last evaluation with a more accurate one
        evaluate(player=player, games=evaluation_games, log_method=log_message, silent=False)
        player.plotter.plot_results(experiment_name=EXPERIMENT_NAME)
        # player.plotter.plot_scores(comment=" e:%s, a:%s" % (e, a))

    log_message("|--- LR:%s Alpha: %s Score: %s Simulation time: %s ---|" % (lr, a, (player1.score+player2.score)/2, str(datetime.now() - start_time).split(".")[0]))

    return (player1.score + player2.score)/2


if __name__ == "__main__":
    learning_rates = [float("1e-%d" %i) for i in range(3, 4)]
    alphas = [float("1e-%d" %i) for i in range(1, 7)]

    TRAINING_GAMES = 30000
    EVALUATION_PERIOD = 300  # How often the performance is evaluated
    EVALUATION_GAMES = 20   # Number of final evaluation games

    start_time = datetime.now()

    results = [(evaluation(TRAINING_GAMES, EVALUATION_PERIOD, EVALUATION_GAMES, lr, a), lr, a) for lr in learning_rates for a in alphas]
    log_message("\n")
    for r in sorted(results):
        log_message("score:%s lr:%s a:%s" % r)

    print("CrossValidation timer: %s" % str(datetime.now() - start_time).split(".")[0])
    evaluation_file.close()
