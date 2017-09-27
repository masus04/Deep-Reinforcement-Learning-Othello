import src.config as config
from src.othello import Othello
from src.player import MCPlayer, TDPlayer

player1 = TDPlayer(color=config.BLACK)
player2 = MCPlayer(color=config.WHITE)

simulation = Othello(player1, player2)

""" Continue training """
# player1.load_params()

""" | Training script | """

TOTAL_GAMES = 200
EVALUATION_GAMES = 0

# training
print("Started training with cuda")
simulation.run_training_simulations(TOTAL_GAMES-EVALUATION_GAMES, cuda=True)

print("Started training without cuda")
simulation.run_training_simulations(TOTAL_GAMES-EVALUATION_GAMES, cuda=False)

""" | Training script | """
