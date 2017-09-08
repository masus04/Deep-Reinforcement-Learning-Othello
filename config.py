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
LABEL_LOSS = 1
LABEL_WIN = 2

LEARNING_RATE = 0.1
MINIBATCH_SIZE = 1

EPSILON = 0.01
EPSILON_REDUCE = .99995

def get_color_from_player_number(number):
    if number == BLACK:
        return "Black"
    else:
        return "White"
