import math

import src.config as config
from src.othello import Othello
from src.player import ReportingPlayer, RandomPlayer, HeuristicPlayer
from src.heuristic import OthelloHeuristic

heur = HeuristicPlayer(color=config.BLACK)

player1 = ReportingPlayer(heur)
player2 = ReportingPlayer(RandomPlayer(color=config.WHITE))

simulation = Othello(player1, player2)


def generate_greedy_data_set(games, silent=True):
    """ Generates a dataset containing pairs of (Board, label) where label is the color of the player
    with more pieces on the board """

    def criterion(board):
        return (board.white_pieces < board.black_pieces) + config.LABEL_LOSS

    return __generate_data_set__(games, criterion, silent)


def generate_heuristic_data_set(games, silent=True):
    """ Generates a dataset containing pairs of (Board, label) where label is the color of the player with the better board according to the heuristic evaluation """

    def criterion(board):
        return (heur.evaluate(board) > 0) + config.LABEL_LOSS

    return __generate_data_set__(games, criterion, silent)


def generate_mobility_data_set(games, silent=True):
    """ Generates a dataset containing pairs of (Board, label) where label is the color of the player with the higher mobility (possible moves) """
    heuristic = OthelloHeuristic(strategy=OthelloHeuristic.PURE_MOBILITY_STRATEGY)

    def criterion(board):
        return (heuristic.evaluate(board, config.BLACK, config.WHITE) > heuristic.evaluate(board, config.WHITE, config.BLACK)) + config.LABEL_LOSS

    return __generate_data_set__(games, criterion, silent)


def generate_save_stones_data_set(games, silent=True):
    """ Generates a dataset containing pairs of (Board, label) where label is the color of the player with control over more save stones (stones that cannot be flipped again)"""
    heuristic = OthelloHeuristic(strategy=OthelloHeuristic.SAVE_STONES_STRATEGY)

    def criterion(board):
        return (heuristic.evaluate(board, config.BLACK, config.WHITE) > heuristic.evaluate(board, config.WHITE, config.BLACK)) + config.LABEL_LOSS

    return __generate_data_set__(games, criterion, silent)


def __generate_data_set__(games, criterion, silent=True):
    """Simulates a number of games between a HeuristicPlayer and a RandomPlayer and applies a given criterion to label the resulting board states"""
    boards = []
    samples = []
    labels = []
    simulation.run_simulations(games, silent=True)
    boards += player1.pop_report() + player2.pop_report()

    for board in boards:
        samples.append(board.board)
        labels.append(criterion(board))

    return samples, labels


if __name__ == "__main__":
    dataset = generate_heuristic_data_set(100, True)
    print(len(dataset[1]))
    print(max(dataset[1]), min(dataset[1]))
