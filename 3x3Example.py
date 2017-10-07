import src.config as config
from src.player import TDPlayer
from src.valueFunction import ThreeByThreeVF
from training import train

player1 = TDPlayer(color=config.BLACK, strategy=ThreeByThreeVF)
player2 = TDPlayer(color=config.WHITE, strategy=ThreeByThreeVF)

TOTAL_GAMES = 1000
EVALUATION_PERIOD = 10

train(player1, player2, TOTAL_GAMES, EVALUATION_PERIOD)

# save artifacts
for player in (player1, player2):
    player.plotter.plot_results()
    player.plotter.plot_scores()
    player.save()

print("Training completed")
