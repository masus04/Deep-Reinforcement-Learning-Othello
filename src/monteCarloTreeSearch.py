import math
from datetime import datetime

import src.config as config


class MCTS:

    def __init__(self, color, board, value_function):
        self.color = color
        self.root = MCTSNode(move=None, board=None, parent=None)
        self.value_function = value_function

        # Generate all possible afterstates, add them as leafs and evaluate & backup
        self.root.children = [MCTSNode(move=move, board=board.copy().apply_move(move, color), parent=self.root) for move in board.get_valid_moves(color)]
        for child in self.root.children:
            child.backup(self.evaluate(child))

    def get_leaf(self, exploration=True):
        node = self.root

        while not node.is_leaf():
            if exploration:
                node = max((child for child in node.children), key=lambda x: x.get_in_tree_score())
            else:
                node = max((child for child in node.children), key=lambda x: x.score / x.visits)
        return node

    def extend_tree(self, time_limit):
        start_time = datetime.now()

        while (datetime.now()-start_time).seconds < time_limit:
            leaf = self.get_leaf()
            leaf.generate_children(self.color)
            for child in leaf.children:
                child.backup(self.evaluate(child))

    def evaluate(self, leaf):
        return self.value_function.evaluate(leaf.board.get_representation(self.color))


class MCTSNode:

    def __init__(self, move, board, parent):
        self.parent = parent
        self.move = move
        self.board = board
        self.children = []
        self.visits = 0
        self.score = 0

    def is_leaf(self):
        return len(self.children) == 0

    def get_in_tree_score(self):  # Factor prior probability =^= actionValue? Cross validate TREE_EXPLORATION constant
        return self.score/self.visits + math.sqrt(self.parent.visits-self.visits)/self.visits * config.TREE_EXPLORATION

    def generate_children(self, color):
        states = (self.board.copy().apply_move(move, config.other_color(color)) for move in self.board.get_valid_moves(config.other_color(color)))  # Opponents afterstates
        afterstates = ((move, state.copy().apply_move(move, color)) for state in states for move in state.get_valid_moves(color))  # Players afterstates
        self.children = [MCTSNode(move=afterstate[0], board=afterstate[1], parent=self) for afterstate in afterstates]

    def backup(self, score):
        self.visits += 1
        self.score += score

        if self.parent:  # This is not the root node -> stop criterion
            self.parent.backup(score)  # recursion
