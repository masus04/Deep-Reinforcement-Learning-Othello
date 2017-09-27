import src.config as config
from src.othello import Othello
from src.player import ComputerPlayer, RandomPlayer, MCPlayer, TDPlayer
from src.plotter import print_inplace
from datetime import datetime

player1 = TDPlayer(color=config.BLACK)
player2 = RandomPlayer(color=config.WHITE)

simulation = Othello(player1, player2)
start_time = datetime.now()

""" Continue training """
# player1.load_params()


def run_simulations(episodes, silent=True):
    sum = 0
    for i in range(episodes):
        result = simulation.run()
        if not silent:
            print("Winner: %s" % config.get_color_from_player_number(result))
        if result == config.BLACK:
            sum += 1
        player1.value_function.plotter.add_result(result)
        print_inplace("Episode %s/%s" % (i+1, episodes), (i+1)/episodes*100, datetime.now()-start_time)

    print()
    return sum/episodes*100


""" | Training script | """

TOTAL_GAMES = 10000
EVALUATION_GAMES = 100

# training
print("Started training")
print("Player 1 won " + str(run_simulations(episodes=TOTAL_GAMES-EVALUATION_GAMES)) + "% of games in training\n")
# evaluation
print("Started evaluation")
print("Player 1 won " + str(run_simulations(episodes=EVALUATION_GAMES)) + "% of games in evaluation")

player1.value_function.plotter.plot_results("MCPlayer: %s Episodes" % TOTAL_GAMES, resolution=100)
player1.save_params()

""" | Training script | """


print("Training took |%s|" % (datetime.now()-start_time))
