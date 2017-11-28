#!/usr/bin/env python3
import math
from datetime import datetime
from statistics import median, stdev, variance

import src.config as config
from src.player import TDPlayer
from src.valueFunction import ValueFunction, SimpleValueFunction, FCValueFunction
from generateDataSet import generate_greedy_data_set, generate_heuristic_data_set
from src.plotter import Printer


def test_with_parameters(games, learning_rate=config.LEARNING_RATE, comment=""):
    start_time = datetime.now()
    test_samples, test_labels = generate_heuristic_data_set(100)

    """ Load ValueFunction """
    player = TDPlayer(config.BLACK, strategy=SimpleValueFunction, lr=learning_rate)

    i = 0
    batch_size = 100
    batches = math.ceil(games/batch_size)
    while i < batches:
        i += 1
        samples, labels = generate_heuristic_data_set(batch_size if games//(i*batch_size) >= 1 else games%batch_size)
        player.plotter.add_loss(player.value_function.update(samples, labels))
        printer.print_inplace("Training batch %s/%s" % (i, batches), 100*i//batches, (str(datetime.now()-start_time)).split(".")[0])
        player.plotter.add_accuracy(evaluate_accuracy(test_samples, test_labels, player.value_function))

    print("Evaluation:")
    player.plotter.plot_accuracy(" lr:{} ".format(learning_rate) + "final score:{0:.3g}".format(player.plotter.accuracies.get_values()[-1]))
    # player.save()


def evaluate_accuracy(samples, labels, value_function, silent=True):
    accuracy_sum = 0
    evaluation_samples = round(len(samples)/10)
    for i in range(evaluation_samples):
        prediction = value_function.evaluate(samples[i])
        accuracy_sum += (prediction > (config.LABEL_WIN - config.LABEL_LOSS)/2) == (labels[i] > (config.LABEL_WIN - config.LABEL_LOSS)/2)
        if not silent:
            print("Sample: %s, Label: %s, Prediction: %s" % (i, labels[i], "{0:.3g}".format(prediction)))
    return accuracy_sum/evaluation_samples


def compare_afterstate_values(value_function):

    test_samples, test_labels = generate_greedy_data_set(250)
    afterstate_values = [value_function.evaluate(sample) for sample in test_samples]

    print("Max:%s Min:%s Median:%s StandardDeviation:%s" % (max(afterstate_values), min(afterstate_values), median(afterstate_values), stdev(afterstate_values)), variance(afterstate_values))
    print("Exact values:")
    for value in afterstate_values:
        print(value)


""" Configure Parameters here, adjust Network in valueFunction.SimpleValueFunction """

printer = Printer()
start_time = datetime.now()

GAMES = 50000

# value_function = config.load_player("TDPlayer_Black_ValueFunction|Async|").value_function
# compare_afterstate_values(value_function)

# test_with_parameters(games=GAMES, learning_rate=float(round(0.1**3.5, 7)))

learning_rates = range(6)
for i, lr in enumerate(learning_rates):
    test_with_parameters(games=GAMES, learning_rate=float(round(0.1**lr, 7)), comment="(%s)" % (i%2))
    print("Simulation time: %s\n" % (str(datetime.now()-start_time)).split(".")[0])

