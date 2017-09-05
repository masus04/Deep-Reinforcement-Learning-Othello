# Constants and configuration options.

TIMEOUT = 0.2

EMPTY = 0
BLACK = 1
WHITE = 2

HEADLESS = True
HUMAN = "Human"
COMPUTER = "Computer"


def get_color_from_player_number(number):
    if number == BLACK:
        return "Black"
    else:
        return "White"
