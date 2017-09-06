import random
import config
from game_ai import GameArtificialIntelligence
from heuristic import OthelloHeuristic
from gui import NoGui
from valueFunction import ValueFunction
from board import Board


class Player(object):

    def __init__(self, color, time_limit=config.TIMEOUT, gui=NoGui()):
        self.color = color
        self.time_limit = time_limit
        self.gui = gui

    def get_move(self, board):
        raise NotImplementedError("function get_move must be implemented by subclass")

    def set_gui(self, gui):
        self.gui = gui
        return self


class HumanPlayer(Player):

    def __init__(self, color, time_limit=-1, gui=None):
        if isinstance(gui, NoGui):
            raise Exception("Human Player cannot be used in headless games")
        super(HumanPlayer, self).__init__(color, time_limit, gui)

    def get_move(self, board):
        valid_moves = board.get_valid_moves(self.color)
        self.gui.highlight_valid_moves(valid_moves)
        while True:
            move = self.gui.get_move_by_mouse()
            if move in valid_moves:
                break
        return move


class RandomPlayer(Player):

    def get_move(self, board):
        return random.sample(board.get_valid_moves(self.color), 1)[0]


class ComputerPlayer(Player):

    def __init__(self, color, time_limit=config.TIMEOUT, gui=NoGui(), strategy=OthelloHeuristic.DEFAULT_STRATEGY):
        super(ComputerPlayer, self).__init__(color, time_limit, gui)
        heuristic = OthelloHeuristic(strategy)
        self.ai = GameArtificialIntelligence(heuristic.evaluate)

    def get_move(self, board):
        other_color = config.BLACK
        if self.color == config.BLACK:
            other_color = config.WHITE

        return self.ai.move_search(board, self.time_limit, self.color, other_color)


class DeepRLPlayer(Player):

    def __init__(self, color, time_limit=config.TIMEOUT, gui=NoGui(), strategy=None):
        super(DeepRLPlayer, self).__init__(color, time_limit, gui)
        self.valueFunction = ValueFunction()

    def get_move(self, board):
        return self.behaviour_policy(board)

    def behaviour_policy(self, board):
        afterstates = [(Board(board.get_representation(self.color)).apply_move(valid_move, config.BLACK), valid_move) for valid_move in board.get_valid_moves(self.color)]
        return max(((self.valueFunction.evaluate(afterstate[0]), afterstate[1]) for afterstate in afterstates))[1]

