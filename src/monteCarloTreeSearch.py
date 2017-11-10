import math
from datetime import datetime
import numpy as np

import src.config as config


class MCTS:
    def __init__(self, color, board, value_function, tree_exploration_constant):
        self.color = color
        self.root = MCTSNode(move=None, board=None, parent=None, value_function=value_function)
        self.root.edges = [MCTSEdge(board=board, color=color, parent=self.root, value_function=value_function)]
        self.depths = []
        self.tree_exploration_constant = tree_exploration_constant

    def calculate_depth(self):
        node = self.root
        depth = 0
        while not node.is_leaf():
            node = self.__get_max_child_node__(node, lambda node: node.score / node.visits)
            depth += 1
        self.depths.append(depth)

    def register_opponents_move(self, board):
        if not self.root.is_leaf():
            try:
                self.root.edges = [find_edge_by_board(self.root.edges, board)]
            except StopIteration:  # debug this
                next_layer_edges = [edge2 for edge in self.root.edges for node in edge.nodes for edge2 in node.edges ]
                self.root.edges = find_edge_by_board(next_layer_edges, board)

                # self.root = MCTSNode(move=None, board=None, parent=None, value_function=None)
                # self.root.edges = [MCTSEdge(board=board, color=self.color, parent=self.root, value_function=self.root.value_function)]

    def get_leaf(self):
        node = self.root

        while not node.is_leaf():
            node = self.__get_max_child_node__(node, self.__get_in_tree_score__)

        return node

    def choose_node(self):
        self.calculate_depth()
        self.root = self.__get_max_child_node__(self.root, lambda node: node.score / node.visits)
        self.root.generate_edges(self.color)
        return self.root

    def extend_tree(self, time_limit):
        start_time = datetime.now()

        while (datetime.now() - start_time).seconds < time_limit:
            leaf = self.get_leaf()
            leaf.generate_edges(self.color)

    @staticmethod
    def __get_max_child_node__(node, criterion):
        return max((node for edge in node.edges for node in edge.nodes), key=criterion)

    def __get_in_tree_score__(self, node):  # Factor prior probability =^= actionValue? Cross validate TREE_EXPLORATION constant
        return node.score / node.visits + math.sqrt(node.parent.parent.visits - node.visits) / node.visits * self.tree_exploration_constant


class MCTSNode:
    def __init__(self, move, board, parent, value_function):
        self.parent = parent
        self.move = move
        self.board = board
        self.edges = []
        self.visits = 0
        self.score = 0
        self.value_function = value_function

    def is_leaf(self):
        return len(self.edges) == 0

    def generate_edges(self, color):
        valid_moves = self.board.get_valid_moves(config.other_color(color))
        # Condition for passes and end of simulation
        if len(valid_moves) == 0:
            winner = self.board.game_won()
            if winner:
                self.value_function = lambda x: config.get_result_label(winner == color)
            else:
                self.edges = [MCTSEdge(board=self.board.copy(), color=color, parent=self, value_function=self.value_function)]
            return

        # Continue regular simulation
        if len(self.edges) == 0:  # Most likely case, this is an optimization
            self.edges = [MCTSEdge(board=self.board.copy().apply_move(move, config.other_color(color)), color=color, parent=self, value_function=self.value_function)
                          for move in valid_moves]  # Opponents afterstates

        elif not (len(self.edges) == len(valid_moves)):  # if lengths equal, do nothing, else insert missing nodes
            for move in valid_moves:
                board = self.board.copy().apply_move(move, config.other_color(color))
                try:
                    find_edge_by_board(self.edges, board)
                except StopIteration:
                    self.edges.append(MCTSEdge(board, color, self, self.value_function))

    def evaluate_and_backup(self, color):
        self.visits += 1
        if self.parent:
            self.score += self.value_function(self.board.get_representation(color))
            self.parent.backup(self.score)  # recursion


class MCTSEdge:
    def __init__(self, board, color, parent, value_function):
        self.board = board
        self.parent = parent
        self.nodes = []
        self.__generate_nodes__(color, value_function)

    def __generate_nodes__(self, color, value_function):
        valid_moves = self.board.get_valid_moves(color)
        # Condition for passing and end of simulation
        if len(valid_moves) == 0:
            winner = self.board.game_won()
            child = MCTSNode(move=None, board=self.board.copy(), parent=self, value_function=(lambda x: config.get_result_label(winner == color)) if winner else value_function)
            self.nodes = [child]
            child.evaluate_and_backup(color)

        # Continue regular simulation
        else:
            afterstates = ((move, self.board.copy().apply_move(move, color)) for move in valid_moves)  # Players afterstates
            for afterstate in afterstates:
                child = MCTSNode(move=afterstate[0], board=afterstate[1], parent=self, value_function=value_function)
                self.nodes.append(child)
                child.evaluate_and_backup(color)

    def backup(self, score):
        self.parent.evaluate_and_backup(score)


def find_edge_by_board(edges, board):
    return next((edge for edge in edges if (edge.board.board == board.board).all()))
