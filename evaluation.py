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

    player.train = False
    temp_e = player.e
    player.e = -1
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

    player.score /= len(reference_players)  # Normalize to 100pts max
    player.train = True
    player.e = temp_e

    player.plotter.add_evaluation_score(player.score)
    if not silent:
        log_method("|-- %s achieved an evaluation score of: %s --|" % (player.player_name, player.score))
    return player.score


if __name__ == "__main__":

    td_black = TDPlayer.load_player(color=config.BLACK, strategy=ValueFunction)
    td_white = TDPlayer.load_player(color=config.WHITE, strategy=ValueFunction)

    # mc_player = MCPlayer.load_player(color=config.BLACK, strategy=ValueFunction)

    for player in [td_black, td_white]:
        evaluate(player)
        player.plotter.plot_results()
        player.plotter.plot_scores()
