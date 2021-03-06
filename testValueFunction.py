#!/usr/bin/env python3
import os
import math
from datetime import datetime
from statistics import median, stdev, variance

import core.config as config
from core.player import TDPlayer
import core.valueFunction as vF
from generateDataSet import generate_greedy_data_set, generate_heuristic_data_set, generate_combined_data_set, generate_save_stones_data_set, generate_mobility_data_set
from core.plotter import Printer

printer = Printer()
STRATEGY = vF.LargeValueFunction
EXPERIMENT_NAME = "testValueFunction %s" % STRATEGY.__name__

if not os.path.exists("./plots/" + EXPERIMENT_NAME):
    os.makedirs("./plots/" + EXPERIMENT_NAME)


evaluation_file = open("./plots/testValueFunction_evaluationfile_%s.txt" % STRATEGY.__name__, "w+")


def log_message(message):
    print(message)
    evaluation_file.write(message + "\n")


def test_with_parameters(games, strategy, labeling_strategy, test_set, learning_rate=config.LEARNING_RATE, comment=""):
    test_samples, test_labels = test_set[0], test_set[1]
    player = TDPlayer(config.BLACK, strategy=strategy, lr=learning_rate)

    i = 0
    batch_size = 100
    batches = math.ceil(games/batch_size)
    start_time = datetime.now()
    while i < batches:
        i += 1
        samples, labels = labeling_strategy(batch_size if games//(i*batch_size) >= 1 else games%batch_size)
        player.plotter.add_loss(player.value_function.update(samples, labels))
        printer.print_inplace("Training batch %s/%s" % (i, batches), 100*i//batches, (str(datetime.now()-start_time)).split(".")[0])
        player.plotter.add_accuracy(evaluate_accuracy(test_samples, test_labels, player.value_function, test_time=True))

    print("Evaluation:")
    player.plotter.plot_accuracy(experiment_name=EXPERIMENT_NAME, comment="labelingStrategy: {}".format(labeling_strategy.__name__) + "final score:{0:.3g}".format(player.plotter.accuracies.get_values()[-1]), path="/" + EXPERIMENT_NAME)
    player.save("_labeling_strategy: %s lr:%s" % (labeling_strategy.__name__, learning_rate))
    return player.plotter.accuracies.get_values()[-1], player


def evaluate_accuracy(samples, labels, value_function, test_time=False, silent=True):
    mapping = round if test_time else id  # At training time, compare with real label value, at test time only expect boolean decision

    accuracy_sum = 0
    evaluation_samples = round(len(samples)/10)
    for i in range(evaluation_samples):
        prediction = value_function.evaluate(samples[i])
        accuracy_sum += mapping(prediction) == mapping(labels[i])
        if not silent:
            print("Sample: %s, Label: %s, Prediction: %s" % (i, labels[i], "{0:.3g}".format(prediction)))
    return accuracy_sum/evaluation_samples


def compare_afterstate_values(value_function, labeling_strategy):

    test_samples, test_labels = labeling_strategy(250)
    afterstate_values = [value_function.evaluate(sample) for sample in test_samples]

    print("Max:%s Min:%s Median:%s StandardDeviation:%s" % (max(afterstate_values), min(afterstate_values), median(afterstate_values), stdev(afterstate_values)), variance(afterstate_values))
    print("Exact values:")
    for value in afterstate_values:
        print(value)


def cross_validation():
    start_time = datetime.now()

    GAMES = 200000

    log_message("Crossvalidation of %s over %s games" % (STRATEGY.__name__, GAMES))

    # compare_afterstate_values(value_function=value_function, labeling_strategy=LABELING_STRATEGY)

    for label_strategy in [generate_mobility_data_set, generate_save_stones_data_set, generate_combined_data_set]:
        log_message("  | --- Labeling strategy: %s --- |  " % label_strategy.__name__)
        results = []

        test_set = label_strategy(100)

        for i, exponent in enumerate(range(1, 3)):
            lr = float(round(0.1 ** exponent, 7))
            results.append((lr, test_with_parameters(games=GAMES, strategy=STRATEGY, labeling_strategy=label_strategy, test_set=test_set, learning_rate=lr)[0]))
            log_message("Simulation time: %s\n" % (str(datetime.now() - start_time)).split(".")[0])

        results.sort()
        log_message("\nAccuracy scores:")
        for result in results:
            log_message("lr: %s, accuracy: %s" % (result[0], result[1]))

    print("\nExperiment completed\n")


def training():

    LABELING_STRATEGY = generate_heuristic_data_set

    accuracies, player = test_with_parameters(games=10000, strategy=vF.LargeValueFunction, labeling_strategy=LABELING_STRATEGY, test_set=LABELING_STRATEGY(500), learning_rate=0.1)


if __name__ == "__main__":

    # cross_validation()
    training()
