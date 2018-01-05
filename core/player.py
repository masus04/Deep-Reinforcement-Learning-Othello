import os
import random
import torch

import core.config as config
from core.game_ai import GameArtificialIntelligence
from core.heuristic import OthelloHeuristic
from core.gui import NoGui
from core.valueFunction import ValueFunction, NoValueFunction
# from core.board import Board
from core.plotter import Plotter, NoPlotter

from core.gridWorld import GridWorldBoard as Board

class Player(object):

    def __init__(self, color, strategy=None, time_limit=config.TIMEOUT, gui=NoGui()):
        self.color = color
        self.player_name = "%s_%s_%s" % (self.__class__.__name__, config.get_color_from_player_number(self.color), strategy.__name__ if strategy else "")
        self.time_limit = time_limit
        self.gui = gui
        self.plotter = NoPlotter()
        self.value_function = NoValueFunction()
        self.train = True
        self.opponents = {}

    def get_move(self, board):
        raise NotImplementedError("function get_move must be implemented by subclass")

    def set_gui(self, gui):
        self.gui = gui
        return self

    def register_winner(self, winner_color):
        pass

    def save(self, comment=""):
        pass

    def set_name(self, name):
        self.player_name = name
        self.plotter.plot_name = self.player_name

    @classmethod
    def load_player(cls, color):
        return cls(color)

    def add_opponent(self, opponent, episodes):
        if self.train:
            if opponent.player_name in self.opponents:
                self.opponents[opponent.player_name] += episodes
            else:
                self.opponents[opponent.player_name] = episodes

    def __generate_afterstates__(self, board):
        """ returns a list of Board instances, one for each valid move. The player is always Black in this representation. """
        return [(Board(board.get_representation(self.color)).apply_move(valid_move, config.BLACK), valid_move) for valid_move in board.get_valid_moves(self.color)]


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


class HeuristicPlayer(Player):
    """ This is a standardized Benchmark player. Good implementations reach up to 95% win rate against it """
    heuristic_table =[[100, -25, 10, 5, 5, 10, -25, 100],
                      [-25, -25,  2, 2, 2, 2,  -25, -25],
                      [ 10,   2,  5, 1, 1, 5,    2,  10],
                      [  5,   2,  1, 2, 2, 1,    2,   5],
                      [  5,   2,  1, 2, 2, 1,    2,   5],
                      [ 10,   2,  5, 1, 1, 5,    2,  10],
                      [-25, -25,  2, 2, 2, 2,  -25, -25],
                      [100, -25, 10, 5, 5, 10, -25, 100]]

    def get_move(self, board):
        afterstates = [[self.evaluate(afterstate[0]), afterstate[1]] for afterstate in self.__generate_afterstates__(board)]
        return max(afterstates)[1]

    def evaluate(self, board):
        board = board.board
        value = 0
        for i in range(len(self.heuristic_table)):
            for j in range(len(self.heuristic_table[0])):
                # afterstates are always from the perspective of the Black player
                if board[i][j] == config.BLACK:
                    value += self.heuristic_table[i][j] * int(board[i][j])
                elif board[i][j] == config.WHITE:
                    value -= self.heuristic_table[i][j] * int(board[i][j])
        return value


class ReportingPlayer:
    """ A wrapper for players that stores all of their afterstates and makes them accessible """

    def __init__(self, player):
        self.player = player
        self.color = player.color
        self.reportedBoards = []
        self.train = False

    @property
    def player_name(self):
        return self.player.player_name

    def get_move(self, board):
        self.player.color = self.color
        move = self.player.get_move(board)
        self.reportedBoards.append(board.copy().apply_move(move, self.color))
        return move

    def set_gui(self, gui):
        self.player.set_gui(gui)
        return self

    def register_winner(self, winner_color):
        return self.player.register_winner(winner_color)

    def add_opponent(self, opponent, episodes):
        return self.player.add_opponent(opponent, episodes)

    def pop_report(self):
        report = self.reportedBoards
        self.reportedBoards = []
        return report


class DeepRLPlayer(Player):

    """ DeepRLPlayers handle the interaction between the game and their value function.
        Inside the player, the Board is represented as a Board object. However only the np.array board is passed to the evaluation function"""

    def __init__(self, color, strategy=ValueFunction, lr=config.LEARNING_RATE, alpha=config.ALPHA, e=config.EPSILON, time_limit=config.TIMEOUT, gui=NoGui()):
        super(DeepRLPlayer, self).__init__(color=color, strategy=strategy, time_limit=time_limit, gui=gui)
        self.e = e
        self.alpha = alpha
        self.plotter = Plotter(self.player_name)
        self.value_function = strategy(learning_rate=lr)
        self.training_samples = []
        self.training_labels = []

    def copy_with_inversed_color(self):
        player = self.__class__(color=config.other_color(self.color), strategy=self.value_function.__class__)
        player.value_function = self.value_function.copy()
        player.plotter = self.plotter.copy()
        player.opponents = self.opponents.copy()
        return player

    def get_move(self, board):
        return self.__behaviour_policy__(board)

    def register_winner(self, winner_color):
        if self.train:
            self.__generate_training_labels__(winner_color)
            self.plotter.add_loss(self.value_function.update(self.training_samples, self.training_labels))
            self.alpha *= config.ALPHA_REDUCE
        self.training_samples = []
        self.training_labels = []

    def __behaviour_policy__(self, board):
        raise NotImplementedError("function behaviour_policy must be implemented by subclass")

    def __e_greedy__(self, lst):
        if random.random() > self.e:
            try:
                result = max(lst)
            except Exception:
                result = random.choice(lst)
        else:
            result = random.choice(lst)

        # self.e = self.e*config.EPSILON_REDUCE  # This is experimental
        return result

    def save(self, comment="", path="/"):
        if not os.path.exists("./Players" + path):
            os.makedirs("./Players" + path)
        torch.save(self, "./Players" + path + "%s%s.pth" % (self.player_name, comment))

    @classmethod
    def load_player(cls, color, strategy):
        """ Loads model to the device it was saved to, except if cuda is not available -> load to cpu """
        player_name = "%s_%s_%s" % (cls.__name__, config.get_color_from_player_number(color), strategy.__name__)
        map_location = None if config.CUDA else lambda storage, loc: storage
        return torch.load("./Players/%s.pth" % player_name, map_location=map_location)

    def save_params(self):
        """  DEPRECATED """
        if not os.path.exists("./Weights"):
            os.makedirs("./Weights")
        torch.save(self.value_function, "./Weights/%s.pth" % self.player_name)


class MCPlayer(DeepRLPlayer):

    def __behaviour_policy__(self, board):
        afterstates = self.__generate_afterstates__(board)
        afterstates = [(self.value_function.evaluate(afterstate[0].board), afterstate[0], afterstate[1]) for afterstate in afterstates]
        if self.train:
            afterstate = self.__e_greedy__(afterstates)
        else:
            afterstate = max(afterstates)
        self.training_samples += [afterstate[1].board]  # Add afterstate board_sample
        return afterstate[2]

    def __generate_training_labels__(self, winner_color):
        self.training_labels = [config.get_result_label(winner_color, self.color) for sample in self.training_samples]


class TDPlayer(MCPlayer):

    def __generate_training_labels__(self, winner_color):
        for i in range(len(self.training_samples)-1):
            self.training_labels.append(self.__td_target__(self.training_samples[i], self.training_samples[i + 1]))
        self.training_labels.append(config.get_result_label(winner_color, self.color))

    def __td_target__(self, state, next_state):
        v_state = self.value_function.evaluate(state)
        v_next_state = self.value_function.evaluate(next_state)
        return v_state + self.alpha * (v_next_state - v_state)


class GridWorldOpponent(Player):

    def get_move(self, board):
        return None
