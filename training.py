import config
from othello import Othello
from player import ComputerPlayer, RandomPlayer, DeepRLPlayer
from heuristic import OthelloHeuristic
from valueFunction import ValueFunction
from datetime import datetime

# player1 = ComputerPlayer(color=config.BLACK, time_limit=config.TIMEOUT, strategy=OthelloHeuristic.DEFAULT_STRATEGY)
# player1 = DeepRLPlayer(color=config.BLACK, time_limit=config.TIMEOUT, strategy=ValueFunction())
# player1 = RandomPlayer(color=config.BLACK, time_limit=config.TIMEOUT)
player1 = DeepRLPlayer(color=config.BLACK, time_limit=config.TIMEOUT, strategy=ValueFunction())
player2 = ComputerPlayer(color=config.WHITE, time_limit=config.TIMEOUT, strategy=OthelloHeuristic.DEFAULT_STRATEGY)

simulation = Othello(player1, player2)
start_time = datetime.now()


""" | Training script | """
episodes = 10

sum = 0
for result in (simulation.run() for i in range(episodes)):
    # print(("Winner: %s" % config.get_color_from_player_number(self.run())))
    if result == config.EMPTY:
        episodes -= 1
    else:
        sum += result
print(sum/episodes)

""" | Training script | """


print("Training took |%s|" % (datetime.now()-start_time))
