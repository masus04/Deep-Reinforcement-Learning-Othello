import src.config as config
from src.valueFunction import ValueFunction, SimpleValueFunction
from src.player import TDPlayer
from src.plotter import Printer
from testValueFunction import evaluate_accuracy
from evaluation import evaluate
from generateDataSet import generate_greedy_data_set, generate_heuristic_data_set, generate_save_stones_data_set, generate_mobility_data_set

import math
from datetime import datetime

EXPERIMENT_NAME = "Supervised_Training"
printer = Printer()


def train_supervised(color, strategy, labeling_strategy, games, learning_rate=config.LEARNING_RATE, experiment_name=EXPERIMENT_NAME):
    test_samples, test_labels = labeling_strategy(100)
    player = TDPlayer(color, strategy=strategy, lr=learning_rate)

    i = 0
    batch_size = 100
    batches = math.ceil(games / batch_size)
    start_time = datetime.now()
    while i < batches:
        i += 1
        samples, labels = labeling_strategy(batch_size if games // (i * batch_size) >= 1 else games % batch_size)
        player.plotter.add_loss(player.value_function.update(samples, labels))
        printer.print_inplace("Training batch %s/%s" % (i, batches), 100 * i // batches, (str(datetime.now() - start_time)).split(".")[0])
        player.plotter.add_accuracy(evaluate_accuracy(test_samples, test_labels, player.value_function))
        evaluate(player, 8, silent=True)

    print("Evaluation:")
    player.plotter.plot_accuracy("labelingStrategy: {} lr:{} ".format(labeling_strategy.__name__, learning_rate) + "final score:{0:.3g}".format(player.plotter.accuracies.get_values()[-1]))
    # player.save()
    return player


if __name__ == "__main__":
    starttime = datetime.now()
    player = train_supervised(config.BLACK, SimpleValueFunction, generate_heuristic_data_set, 500000)
    evaluate(player, 20)

    player.plotter.plot_accuracy(EXPERIMENT_NAME)
    player.plotter.plot_results(EXPERIMENT_NAME)
    player.plotter.plot_scores(EXPERIMENT_NAME)
    player.save(comment=EXPERIMENT_NAME)

    print("Training finished, took %s" % ((str(datetime.now() - starttime)).split(".")[0]))