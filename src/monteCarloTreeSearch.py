from random import random, sample

from src.board import Board


class MCTS:

    EPSILON = 0.1

    def __init__(self, color, board):
        self.color = color
        self.root = MCTSLeaf(move=None, board=None)

        # Generate all possible afterstates and add them as leafs
        self.root.children.append([MCTSLeaf(move, self.root.board.copy().apply_move(move, color)) for move in self.root.board.get_valid_moves(color)])

    def get_leaf(self, e=EPSILON):
        node = self.root

        while not node.is_leaf:
            if random() >= e:
                node = self.max(node.children)  # Apply tree policy
            else:
                move = sample(node.board.get_valid_moves())  # Apply random move and create Node if it does not yet exist
                child = node.get_child_by_move(move)
                if child:
                    node = child
                else:
                    node.children.append(MCTSLeaf(move, node.board.copy().apply_move(self.color)))

        return node

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
