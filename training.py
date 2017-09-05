from othello import Othello
from player import ComputerPlayer


simulation = Othello(ComputerPlayer, ComputerPlayer)

for i in range(5):
    simulation.print_winner()
