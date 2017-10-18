import math

import src.config as config
from src.othello import Othello
from src.player import ReportingPlayer, RandomPlayer, HeuristicPlayer

heur1 = HeuristicPlayer(color=config.BLACK)

player1 = ReportingPlayer(heur1)
player2 = ReportingPlayer(RandomPlayer(color=config.WHITE))

simulation = Othello(player1, player2)


def generate_greedy_data_set(games, silent=True):
    """ Generates a dataset containing pairs of (Board, label) where label is the color of the player
    with more pieces on the board """
    boards = []
    samples = []
    labels = []
    for i in range(games//4):
        if not silent:
            print("Running simulation no. %s" % i)
        simulation.run_simulations(4, silent=True)
        boards += player1.pop_report() + player2.pop_report()

    for board in boards:
        samples.append(board.board)
        labels.append((board.white_pieces < board.black_pieces) + config.LABEL_LOSS)

    return samples, labels


def generate_heuristic_data_set(games, silent=True):
    """ Generates a dataset containing pairs of (Board, label) where label is the color of the player
    with the better board according to the heuristic evaluation """

    """ Generates a dataset containing pairs of (Board, label) where label is the color of the player
    with more pieces on the board """
    boards = []
    samples = []
    labels = []
    for i in range(games//4):
        if not silent:
            print("Running simulation no. %s" % i)
        simulation.run_simulations(4, silent=True)
        boards += player1.pop_report() + player2.pop_report()

    for board in boards:
        samples.append(board.board)
        x = heur1.evaluate(board)
        labels.append(x/1360 + config.LABEL_LOSS)

    return samples, labels


if __name__ == "__main__":
    dataset = generate_heuristic_data_set(20, True)
    print(dataset[1])
    print(max(dataset[1]), min(dataset[1]))
