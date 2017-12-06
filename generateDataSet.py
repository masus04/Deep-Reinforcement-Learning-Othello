import math
from scipy.special import expit

import src.config as config
from src.othello import Othello
from src.player import ReportingPlayer, RandomPlayer, HeuristicPlayer
from src.heuristic import OthelloHeuristic

heuristic_player = HeuristicPlayer(color=config.BLACK)

heuristic = OthelloHeuristic(strategy=OthelloHeuristic.DEFAULT_STRATEGY)

player1 = ReportingPlayer(heuristic_player)
player2 = ReportingPlayer(RandomPlayer(color=config.WHITE))

simulation = Othello(player1, player2)


def generate_greedy_data_set(games, silent=True):
    """ Generates a dataset containing pairs of (Board, label) where label is the color of the player
    with more pieces on the board """

    def criterion(board):
        return board.black_pieces - board.white_pieces

    return __generate_data_set__(games, criterion, silent)


def generate_heuristic_data_set(games, silent=True):
    """ Generates a dataset containing pairs of (Board, label) where label is the color of the player with the better board according to the heuristic evaluation """

    def criterion(board):
        return heuristic_player.evaluate(board)

    return __generate_data_set__(games, criterion, silent)


def generate_mobility_data_set(games, silent=True):
    """ Generates a dataset containing pairs of (Board, label) where label is the color of the player with the higher mobility (possible moves) """

    def criterion(board):
        return heuristic.evaluate_mobility(board, config.BLACK, config.WHITE, 0) - heuristic.evaluate_mobility(board, config.WHITE, config.BLACK, 0)

    return __generate_data_set__(games, criterion, silent)


def generate_save_stones_data_set(games, silent=True):
    """ Generates a dataset containing pairs of (Board, label) where label is the color of the player with control over more save stones (stones that cannot be flipped again)"""

    def criterion(board):
        return heuristic.evaluate_save_stones(board, config.BLACK, config.WHITE, 0) - heuristic.evaluate_save_stones(board, config.WHITE, config.BLACK, 0)

    return __generate_data_set__(games, criterion, silent)


def __generate_data_set__(games, criterion, silent):
    """Simulates a number of games between a HeuristicPlayer and a RandomPlayer and applies a given criterion to label the resulting board states"""
    boards = []
    samples = []
    labels = []
    simulation.run_simulations(games, silent=silent)
    boards += player1.pop_report() + player2.pop_report()

    for board in boards:
        samples.append(board.board)
        score = criterion(board)
        try:
            labels.append(expit(score))
        except Exception as e:
            labels.append(1 if score > 0 else 0)

    return samples, labels


if __name__ == "__main__":
    dataset = generate_save_stones_data_set(100, True)
    print(len(dataset[1]))
    print(max(dataset[1]), min(dataset[1]))
