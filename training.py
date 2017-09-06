from config import BLACK, WHITE, TIMEOUT
from othello import Othello
from player import ComputerPlayer, RandomPlayer, DeepRLPlayer
from heuristic import OthelloHeuristic


player1 = ComputerPlayer(color=BLACK, time_limit=TIMEOUT, strategy=OthelloHeuristic.DEFAULT_STRATEGY)
player2 = RandomPlayer(color=WHITE, time_limit=TIMEOUT)
simulation = Othello(player1, player2)

for i in range(5):
    simulation.print_winner()
