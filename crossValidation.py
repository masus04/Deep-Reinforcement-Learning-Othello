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


def evaluation(games, evaluation_period, evaluation_games, e, a):
    log_message("Evaluating epsilon:%s alpha:%s" % (e, a))

    player1 = PLAYER(color=config.BLACK, strategy=ValueFunction, e=e, alpha=a)
    player2 = PLAYER(color=config.WHITE, strategy=ValueFunction, e=e, alpha=a)
    players = [player1, player2]

    """ Training """

    training.train_and_evaluate(player1=player1, player2=player2, evaluation_period=evaluation_period, games=games)

    """ Evaluation """

    for player in players:
        player.train = False
        player.score = 0
        player.plotter.evaluation_scores = player.plotter.evaluation_scores.get_values()[:-1]  # Replace last evaluation with a more accurate one
        evaluate(player=player, games=evaluation_games, log_method=log_message, silent=False)
        player.plotter.plot_results(comment=" e:%s, a:%s" % (e, a))
        # player.plotter.plot_scores(comment=" e:%s, a:%s" % (e, a))

    log_message("|--- Epsilon:%s Alpha: %s Score: %s Simulation time: %s ---|\n" % (e, a, (player1.score+player2.score)/2, str(datetime.now() - start_time).split(".")[0]))

    return (player1.score + player2.score)/2


if __name__ == "__main__":

    start_time = datetime.now()

    epsilons = [0.001, 0.0003, 0.0001, 0.00003, 0.00001]
    alphas = [0.03, 0.01, 0.003, 0.001]

    TRAINING_GAMES = 200000     # Total training games per configuration
    EVALUATION_PERIOD = 1000    # How often the performance is evaluated
    EVALUATION_GAMES = 80       # Number of final evaluation games

    results = [(evaluation(TRAINING_GAMES, EVALUATION_PERIOD, EVALUATION_GAMES, e, a), e, a) for e in epsilons for a in alphas]
    for r in sorted(results):
        log_message("\nscore:%s e:%s a:%s" % r)

    print("CrossValidation timer: %s" % str(datetime.now() - start_time).split(".")[0])
    evaluation_file.close()
