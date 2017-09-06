import config
from othello import Othello
from player import ComputerPlayer, RandomPlayer, MCPlayer
from heuristic import OthelloHeuristic
from valueFunction import ValueFunction
from datetime import datetime

# player1 = ComputerPlayer(color=config.BLACK, time_limit=config.TIMEOUT, strategy=OthelloHeuristic.DEFAULT_STRATEGY)
# player1 = DeepRLPlayer(color=config.BLACK, time_limit=config.TIMEOUT, strategy=ValueFunction())
# player1 = RandomPlayer(color=config.BLACK, time_limit=config.TIMEOUT)
player1 = MCPlayer(color=config.BLACK, time_limit=config.TIMEOUT, strategy=ValueFunction())
player2 = RandomPlayer(color=config.WHITE, time_limit=config.TIMEOUT)

simulation = Othello(player1, player2)
start_time = datetime.now()


def run_simulations(episodes, print_winner=False):
    sum = 0
    for result in (simulation.run() for i in range(episodes)):
        if print_winner: print("Winner: %s" % config.get_color_from_player_number(result))
        if result == config.EMPTY:
            episodes -= 1
        else:
            sum += result

    return sum/episodes


""" | Training script | """

# training
print("Average result of simulation: %s" % (run_simulations(episodes=100, print_winner=False)))
# evaluation
print("Average result of simulation: %s" % (run_simulations(episodes=10, print_winner=True)))

""" | Training script | """


print("Training took |%s|" % (datetime.now()-start_time))
