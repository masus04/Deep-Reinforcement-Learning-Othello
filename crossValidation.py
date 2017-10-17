from datetime import datetime

import src.config as config
from src.othello import Othello
from src.player import HeuristicPlayer, ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer
from src.valueFunction import ValueFunction, SimpleValueFunction, FCValueFunction
import training
from evaluation import evaluate

start_time = datetime.now()

learning_rates = [float("1e-%d" %i) for i in range(5, 7)]
alphas =         [float("1e-%d" %i) for i in range(1, 7)]

TRAINING_GAMES = 50000
EVALUATION_PERIOD = 250  # How often the performance is evaluated
EVALUATION_GAMES = 100   # Number of final evaluation games
PLAYER = TDPlayer

evaluation_file = open("./plots/crossEvaluation_%s.txt" % PLAYER.__name__, "w+")


def log_message(message):
    print(message)
    evaluation_file.write(message + "\n")


def evaluation(lr, a):
    log_message("\nEvaluating LR:%s a:%s" % (lr, a))

    player1 = PLAYER(color=config.BLACK, strategy=ValueFunction, lr=lr, alpha=a)
    player2 = PLAYER(color=config.WHITE, strategy=ValueFunction, lr=lr, alpha=a)
    players = [player1, player2]

    """ Training """

    training.train(player1=player1, player2=player2, games=TRAINING_GAMES, evaluation_period=EVALUATION_PERIOD)

    """ Evaluation """

    for player in players:
        player.train = False
        player.score = 0
        player.plotter.evaluation_scores = player.plotter.evaluation_scores[:-1]  # Replace last evaluation with a more accurate one
        evaluate(player=player, games=EVALUATION_GAMES, log_method=log_message, silent=False)
        player.plotter.plot_results(comment=" lr:%s, a:%s" % (lr, a))
        player.plotter.plot_scores(comment=" lr:%s, a:%s" % (lr, a))

    log_message("|--- LR:%s Alpha: %s Score: %s Simulation time: %s ---|" % (lr, a, (player1.score+player2.score)/2, str(datetime.now() - start_time).split(".")[0]))

    return (player1.score + player2.score)/2


# results = [(evaluation(lr, a), lr, a) for lr in learning_rates for a in alphas]
results = [(evaluation(lr, lr), lr, lr) for lr in learning_rates]
log_message("\n")
for r in sorted(results):
    log_message("score:%s lr:%s a:%s" % r)

print("CrossValidation timer: %s" % str(datetime.now() - start_time).split(".")[0])
evaluation_file.close()
