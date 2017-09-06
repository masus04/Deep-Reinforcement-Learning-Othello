import config
from othello import Othello
from player import ComputerPlayer, RandomPlayer, DeepRLPlayer
from heuristic import OthelloHeuristic
from valueFunction import ValueFunction
from datetime import datetime

# player1 = ComputerPlayer(color=BLACK, time_limit=TIMEOUT, strategy=OthelloHeuristic.DEFAULT_STRATEGY)
# player1 = DeepRLPlayer(color=BLACK, time_limit=TIMEOUT, strategy=ValueFunction())
player1 = DeepRLPlayer(color=config.BLACK, time_limit=config.TIMEOUT, strategy=ValueFunction())
player2 = RandomPlayer(color=config.WHITE, time_limit=config.TIMEOUT)

simulation = Othello(player1, player2)
start_time = datetime.now()


""" | Training script | """
episodes = 50

sum = 0
for result in (simulation.run() for i in range(episodes)):
    print(config.get_color_from_player_number(result))
    sum = sum+result if result != 0 else sum
print(sum/episodes)

""" | Training script | """


print("Training took |%s|" % (datetime.now()-start_time))
