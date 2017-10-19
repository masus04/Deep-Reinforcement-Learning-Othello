#!/usr/bin/env python3
import os
import torch
from datetime import datetime

import src.config as config
from src.valueFunction import ValueFunction, SimpleValueFunction, FCValueFunction
from generateDataSet import generate_greedy_data_set, generate_heuristic_data_set
from src.plotter import Printer
from src.plotter import Plotter


def test_with_parameters(games, training_episodes, learning_rate=config.LEARNING_RATE, comment=""):
    plot_name = "Heuristic ReLU 7Layers g:%s ep:%s lr:%s - %s %s" % (games, training_episodes, learning_rate, ValueFunction.__name__, comment)
    plotter = Plotter(plot_name)
    printer = Printer()
    test_samples, test_labels = generate_heuristic_data_set(100)
    start_time = datetime.now()

    value_function = ValueFunction(plotter=plotter, learning_rate=learning_rate)
    for i in range(training_episodes):
        samples, labels = generate_heuristic_data_set(games)
        printer.print_inplace("Training Episode %s/%s" % (i+1, training_episodes), (i+1)/training_episodes*100, datetime.now()-start_time)
        plotter.add_accuracy(evaluate_accuracy(test_samples, test_labels, value_function))
        value_function.update(samples, labels)

    print("Evaluation:")
    evaluate_accuracy(test_samples, test_labels, value_function, silent=True)
    print("Training %s episodes for %s games took %s" % (training_episodes, games, datetime.now()-start_time))
    print("Final accuracy: %s\n" % plotter.accuracies[-1])
    plotter.plot_accuracy(" final score:" + "{0:.3g}".format(plotter.accuracies[-1]))
    if not os.path.exists("./weights"):
        os.makedirs("./weights")
    torch.save(value_function, "./weights/%s.pth" % plot_name)


def evaluate_accuracy(samples, labels, value_function, silent=True):
    accuracy_sum = 0
    evaluation_samples = round(len(samples)/10)
    for i in range(evaluation_samples):
        prediction = value_function.evaluate(samples[i])
        accuracy_sum += (prediction > (config.LABEL_WIN - config.LABEL_LOSS)/2) == (labels[i] > (config.LABEL_WIN - config.LABEL_LOSS)/2)
        if not silent:
            print("Sample: %s, Label: %s, Prediction: %s" % (i, labels[i], "{0:.3g}".format(prediction)))
    return accuracy_sum/evaluation_samples


""" Configure Parameters here, adjust Network in valueFunction.SimpleValueFunction """

#test_with_parameters(games=100, training_episodes=500, learning_rate=float(round(0.1**5, 7)))

for i, lr in enumerate([1, 2, 3, 4, 5]):
    test_with_parameters(games=100, training_episodes=500, learning_rate=float(round(0.1**lr, 7)), comment="(%s)" % (i%2))
