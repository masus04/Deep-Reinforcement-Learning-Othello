import src.config as config
from src.othello import Othello
from src.player import HeuristicPlayer, ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer
from src.valueFunction import ValueFunction, SimpleValueFunction, FCValueFunction
from datetime import datetime

start_time = datetime.now()

learning_rates = [float("1e-%d" %i) for i in range(1, 6)]
alphas =         [float("1e-%d" %i) for i in range(1, 6)]

TRAINING_GAMES = 1500
EVALUATION_GAMES = 50

Player = TDPlayer
evaluation_file = open("./plots/crossEvaluation_%s.txt" % Player.__name__, "w")


def log_message(message):
    print(message)
    evaluation_file.write(message + "\n")


def evaluation(lr, a):
    log_message("\nEvaluating LR:%s a:%s" % (lr, a))

    player1 = Player(color=config.BLACK, strategy=ValueFunction, lr=lr, alpha=a)
    player2 = Player(color=config.WHITE, strategy=ValueFunction, lr=lr, alpha=a)

    players = [player1, player2]
    """ Training """
    simulation = Othello(player1, player2)
    simulation.run_simulations(TRAINING_GAMES)

    player1.plotter.plot_results(resolution=200, comment=" lr:%s, a:%s" % (lr, a))
    player2.plotter.plot_results(resolution=200, comment=" lr:%s, a:%s" % (lr, a))

    """ Evaluation """
    ref_players = [[HeuristicPlayer(config.WHITE), RandomPlayer(config.WHITE)],
                   [HeuristicPlayer(config.BLACK), RandomPlayer(config.BLACK)]]

    for player in (players + ref_players[0] + ref_players[1]):
        player.train = False
        player.score = 0

    for i, player in enumerate(players):
        for ref in ref_players[i]:
            simulation = Othello(player, ref)
            results = simulation.run_simulations(EVALUATION_GAMES)
            score = round((sum(results) / EVALUATION_GAMES) * 100)
            player.score += score
            ref.score += 100-score
            log_message("%s won %s of games against %s" % (player.player_name, str(score) + "%", ref.player_name))

        log_message("%s achieved a score of %s" % (player.player_name, player.score))
    print("Simulation time: %s" % str(datetime.now() - start_time).split(".")[0])

    return player1.score + player2.score


results = [(evaluation(lr, a), lr, a) for lr in learning_rates for a in alphas]
log_message("\n")
for r in sorted(results):
    log_message("score:%s lr:%s a:%s" % r)

print(str(datetime.now() - start_time).split(".")[0])
evaluation_file.close()
