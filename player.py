import os
import random
import torch

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

    def register_winner(self, winner_color):
        pass


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

    """ DeepRLPlayers handle the interaction between the game and their value function.
        Inside the player, the Board is represented as a Board object. However only the np.array board is passed to the evaluation function"""

    def __init__(self, color, time_limit=config.TIMEOUT, gui=NoGui(), strategy=None):
        super(DeepRLPlayer, self).__init__(color, time_limit, gui)
        self.valueFunction = ValueFunction()
        self.training_samples = []
        self.training_labels = []

    def get_move(self, board):
        return self.__behaviour_policy__(board)

    def register_winner(self, winner_color):
        raise NotImplementedError("function register_winner must be implemented by subclass")

    def save_params(self):
        if not os.path.exists("./Weights"):
            os.makedirs("./Weights")
        torch.save(self.valueFunction, "./Weights/%s.pth" % self.__class__.__name__)

    def __generate_afterstates__(self, board):
        """ returns a list of Board instances, one for each valid move. The player is always Black in this representation. """
        return [(Board(board.get_representation(self.color)).apply_move(valid_move, config.BLACK), valid_move) for valid_move in board.get_valid_moves(self.color)]

    def __behaviour_policy__(self, board):
        raise NotImplementedError("function behaviour_policy must be implemented by subclass")


class MCPlayer(DeepRLPlayer):

    def __behaviour_policy__(self, board):
        afterstates = self.__generate_afterstates__(board)
        afterstate = max(((self.valueFunction.evaluate(afterstate[0].board), afterstate[0], afterstate[1]) for afterstate in afterstates))
        self.training_samples += [afterstate[1].board]  # Add afterstate board_sample
        return afterstate[2]

    def register_winner(self, winner_color):
        self.training_labels = [config.LABEL_WIN if (self.color == winner_color) else config.LABEL_LOSS for sample in self.training_samples]
        self.valueFunction.update(self.training_samples, self.training_labels)
        self.training_labels = []
        self.training_labels = []
