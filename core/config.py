import os
import sys
import torch

if not os.path.exists("./plots"):
    os.makedirs("./plots")

# Constants and configuration options.

CUDA = "cuda" in sys.argv
print("CUDA %s" % ("enabled" if CUDA else "disabled"))

TIMEOUT = 0.2

# Caution: Adjust those in functions.c as well
EMPTY = 1.5
BLACK = 1
WHITE = 2

HEADLESS = True
HUMAN = "Human"
COMPUTER = "Computer"

# LABEL_LOSS < prediction < LABEL_WIN
LABEL_LOSS = 0
LABEL_WIN = 1
LABEL_DRAW = (LABEL_WIN + LABEL_LOSS) / 2

# Hyperparameters
LEARNING_RATE = 0.05
MINIBATCH_SIZE = 32

ALPHA = 0.003
ALPHA_REDUCE = 0.99999

EPSILON = 0.001
EPSILON_REDUCE = 0.999995


def get_result_label(winner_color, player_color):
    if winner_color == player_color:
        return LABEL_WIN
    if winner_color == other_color(player_color):
        return LABEL_LOSS
    if winner_color == EMPTY:
        return LABEL_DRAW


def get_color_from_player_number(number):
    if number == BLACK:
        return "Black"
    else:
        return "White"


def other_color(color):
    if color == EMPTY:
        return EMPTY
    else:
        return WHITE if color == BLACK else BLACK


def load_player(filename):
    """ loads model to the device it was saved to, except if cuda is not available -> load to cpu """
    map_location = None if CUDA else lambda storage, loc: storage
    return torch.load("./Players/%s.pth" % filename, map_location=map_location)
