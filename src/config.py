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
LEARNING_RATE = float(round(0.1**3, 7))
MINIBATCH_SIZE = 1

EPSILON = 0.01
EPSILON_REDUCE = .99995

ALPHA = 1


def get_color_from_player_number(number):
    if number == BLACK:
        return "Black"
    else:
        return "White"