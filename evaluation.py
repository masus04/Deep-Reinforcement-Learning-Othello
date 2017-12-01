import sys
import src.config as config
from src.othello import Othello
from src.player import ComputerPlayer, RandomPlayer, HeuristicPlayer, MCPlayer, TDPlayer
from src.valueFunction import ValueFunction, SimpleValueFunction

EVALUATION_GAMES = 100


def evaluate(player, games=EVALUATION_GAMES, log_method=print, silent=False):
    other_color = config.WHITE if player.color == config.BLACK else config.BLACK

    heuristic_player = HeuristicPlayer(color=other_color)
    random_player = RandomPlayer(color=other_color)

    reference_players = [heuristic_player, random_player]

    training_flag = player.train
    player.train = False
    player.score = 0

    if not silent:
        log_method("Evaluating %s:" % player.player_name)
    for reference_player in reference_players:
        reference_player.train = False
        simulation = Othello(player, reference_player)
        results = simulation.run_simulations(games, silent=silent)
        player.score += round((sum(results) / games) * 100)
        if not silent:
            log_method("%s won %s of games against %s" % (player.player_name, "{0:.3g}".format((sum(results)/games) * 100) + "%", reference_player.player_name))

    player.train = training_flag

    player.score /= len(reference_players)  # Normalize to 100pts max
    player.plotter.add_evaluation_score(player.score)
    if not silent:
        log_method("|-- %s achieved an evaluation score of: %s --|" % (player.player_name, player.score))
    return player.score


def compare_players(player1, player2, games=EVALUATION_GAMES, silent=False):
    training_flags = player1.train, player2.train

    for player in player1, player2:
        player.train = False

    simulation = Othello(player1, player2)
    results = simulation.run_simulations(games, silent=silent)
    player1.score = sum(results)
    player2.score = games*config.LABEL_WIN - player1.score

    player1.train, player2.train = training_flags

    if not silent:
        print("%s won %s of games against %s" % (player1.player_name, "{0:.3g}".format(player1.score*100/(games*config.LABEL_WIN)) + "%", player2.player_name))

    return (player1.score - player2.score)/2


def train_black_supervised_save_stones_player(games):
    from generateDataSet import generate_save_stones_data_set
    from testValueFunction import test_with_parameters

    testset = generate_save_stones_data_set(100)

    return test_with_parameters(games=games, strategy=SimpleValueFunction, labeling_strategy=generate_save_stones_data_set, test_set=testset, learning_rate=0.1)[1]


if __name__ == "__main__":

    # td_black = config.load_player("TDPlayer_Black_ValueFunction|Async|")
    # td_white = config.load_player("TDPlayer_White_ValueFunction_BEST|Async|")

    td_black = config.load_player("TDPlayer_Black_ValueFunction|Continuous|")
    td_white = config.load_player("TDPlayer_White_ValueFunction|Continuous|")

    assert td_black.color == config.BLACK
    assert td_white.color == config.WHITE

    # Just out of curiosity:

    td_black = train_black_supervised_save_stones_player(10000)

    # mc_player = MCPlayer.load_player(color=config.BLACK, strategy=ValueFunction)

    print(compare_players(player1=td_black, player2=td_white, games=100))

    for player in [td_black, td_white]:
        evaluate(player, games=100)
        # player.plotter.plot_results()
        # player.plotter.plot_scores()
