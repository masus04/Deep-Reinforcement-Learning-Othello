from random import random, sample

from src.board import Board


class MCTS:

    EPSILON = 0.01

    def __init__(self, color, board):
        self.color = color
        self.root = MCTSLeaf(move=None, board=board)

    def get_leaf(self, e=EPSILON):
        node = self.root

        while not node.is_leaf:
            if random() >= e:
                node = self.max(node.children)
            else:
                move = sample(node.board.get_valid_moves())
                child = node.get_child_by_move(move)
                if child:
                    node = child
                else:
                    node.children.append(MCTSLeaf(move, node.board.copy().apply_move(self.color)))
        return node

    def rollout(self, e):
        pass

    @staticmethod
    def max(children):  # Adjust for ties
        candidate = MCTSLeaf(None, None)
        for child in children:
            if child.get_score() > candidate.get_score():
                candidate = child
        return candidate


class MCTSLeaf:

    def __init__(self, move, board):
        self.move = move
        self.board = board
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0

    def get_score(self):
        pass

    def get_child_by_move(self, move):
        for child in self.children:
            if child.move == move:
                return child
        return None
