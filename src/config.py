import os
import torch

if not os.path.exists("./plots"):
    os.makedirs("./plots")

# Constants and configuration options.

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

# Hyperparameters
LEARNING_RATE = 1e-4
MINIBATCH_SIZE = 1

EPSILON = 0.05
EPSILON_REDUCE = 0.99995

ALPHA = 1e-3
ALPHA_REDUCE = 0.99995


def get_color_from_player_number(number):
    if number == BLACK:
        return "Black"
    else:
        return "White"


def load_player(filename):
    """ loads model to the device it was saved to, except if cuda is not available -> load to cpu """
    map_location = None if torch.cuda.is_available() else lambda storage, loc: storage
    return torch.load("./Players/%s.pth" % filename, map_location=map_location)
