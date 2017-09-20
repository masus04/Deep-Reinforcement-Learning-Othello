import src.config as config
from src.othello import Othello
from src.player import ReportingPlayer, RandomPlayer

player1 = ReportingPlayer(RandomPlayer(color=config.BLACK, time_limit=config.TIMEOUT))
player2 = ReportingPlayer(RandomPlayer(color=config.WHITE, time_limit=config.TIMEOUT))

simulation = Othello(player1, player2)


def generate_greedy_data_set(games, silent=True):
    """ Generates a dataset containing pairs of (Board, label) where label is the color of the player
    with more pieces on the board """
    boards = []
    samples = []
    labels = []
    for i in range(games):
        if not silent:
            print("Running simulation no. %s" % i)
        simulation.run()
        boards += player1.pop_report() + player1.pop_report()

    for board in boards:
        samples.append(board.board)
        labels.append((board.white_pieces < board.black_pieces) + config.LABEL_LOSS)

    return samples, labels


def generate_heuristic_data_set(games, silent=True):
    """ Generates a dataset containing pairs of (Board, label) where label is the color of the player
    with the better board according to the heuristic evaluation """
