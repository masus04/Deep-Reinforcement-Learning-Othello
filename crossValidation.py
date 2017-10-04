import src.config as config
from src.othello import Othello
from src.player import HeuristicPlayer, ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer
from src.valueFunction import SimpleValueFunction, FCValueFunction

learning_rates = [float(round(0.1**i, 7)) for i in range(1, 7)]
alphas = [float(round(0.1**i, 8)) for i in range(1, 9)]

TRAINING_GAMES = 10
EVALUATION_GAMES = 10


def evaluation(lr, a):
    player1 = TDPlayer(color=config.BLACK, strategy=SimpleValueFunction, lr=lr, alpha=a)
    player2 = TDPlayer(color=config.WHITE, strategy=SimpleValueFunction, lr=lr, alpha=a)

    players = [player1, player2]
    """ Training """
    simulation = Othello(player1, player2)
    simulation.run_simulations(TRAINING_GAMES)

    player1.plotter.plot_results(comment=" lr:%s, a:%s" % (lr, a))

    """ Evaluation """
    ref_players = [[player2, HeuristicPlayer(config.WHITE), RandomPlayer(config.WHITE)],
                   [HeuristicPlayer(config.BLACK), RandomPlayer(config.BLACK)]]

    for i, player in enumerate(players):
        player.train = False

        for ref in ref_players[i]:
            ref.train = False
            simulation = Othello(player, ref)
            results = simulation.run_simulations(EVALUATION_GAMES)

            print("%s won %s of games against %s\n" % (player.player_name, "{0:.3g}".format((sum(results) / EVALUATION_GAMES) * 100) + "%", ref.player_name))


for lr in learning_rates:
    for a in alphas:
        evaluation(lr, a)
