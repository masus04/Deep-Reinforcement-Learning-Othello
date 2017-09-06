import random
from config import BLACK, WHITE, HEADLESS
from game_ai import GameArtificialIntelligence
from heuristic import OthelloHeuristic
from gui import NoGui
from valueFunction import ValueFunction


class Player(object):

    def __init__(self, color, time_limit=-1, gui=None, strategy=None):
        self.color = color
        self.time_limit = time_limit
        self.gui = gui

    def get_move(self, board):
        raise NotImplementedError("function get_move must be implemented by subclass")

    def set_time_limit(self, new_limit):
        self.time_limit = new_limit


class HumanPlayer(Player):

    def __init__(self, color, time_limit=-1, gui=None, strategy=None):
        if isinstance(gui, NoGui):
            raise Exception("Human Player cannot be used in headless games")
        super(ComputerPlayer, self).__init__(color, time_limit, gui, strategy)

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

    def __init__(self, color, time_limit=5, gui=None, strategy=OthelloHeuristic.DEFAULT_STRATEGY):
        super(ComputerPlayer, self).__init__(color, time_limit, gui, strategy)
        heuristic = OthelloHeuristic(strategy)
        self.ai = GameArtificialIntelligence(heuristic.evaluate)

    def get_move(self, board):
        other_color = BLACK
        if self.color == BLACK:
            other_color = WHITE

        return self.ai.move_search(board, self.time_limit, self.color, other_color)


class DeepRLPlayer(Player):

    def __init__(self, color, time_limit=5, gui=None, strategy=None):
        super(DeepRLPlayer, self).__init__(color, time_limit, gui, strategy)
        self.valueFunction = ValueFunction()

    def get_move(self, board):
        pass
        # TODO: continue here
