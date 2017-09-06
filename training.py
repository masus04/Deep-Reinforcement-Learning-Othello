from othello import Othello
from player import ComputerPlayer, RandomPlayer, DeepRLPlayer


simulation = Othello(ComputerPlayer, RandomPlayer)

for i in range(5):
    simulation.print_winner()
