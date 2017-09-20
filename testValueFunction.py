#!/usr/bin/env python3
import src.config as config
from src.valueFunction import ValueFunction, SimpleValueFunction, FCValueFunction
from generateDataSet import generate_greedy_data_set
from src.plotter import print_inplace
from src.plotter import Plotter
from datetime import datetime


def test_with_parameters(games, training_episodes, learning_rate=config.LEARNING_RATE, plot_name="unnamed"):
    plotter = Plotter()
    test_samples, test_labels = generate_greedy_data_set(10)
    start_time = datetime.now()

    value_function = FCValueFunction(plotter=plotter, learning_rate=learning_rate)
    for i in range(training_episodes):
        samples, labels = generate_greedy_data_set(games)
        print_inplace("Training episode no. %s/%s. Time passed: %s" % (i+1, training_episodes, datetime.now()-start_time))
        plotter.add_accuracy(evaluate_accuracy(test_samples, test_labels, value_function))
        value_function.update(samples, labels)
    print()  # in order not to overwrite inplace

    print("Evaluation:")
    evaluate_accuracy(test_samples, test_labels, value_function, silent=True)
    print("Training %s episodes for %s games took %s" % (training_episodes, games, datetime.now()-start_time))
    print("Final accuracy: %s\n" % plotter.accuracies[-1])
    plotter.plot("%s, %sGames, %sEpisodes, LRate:%s, Accuracy: %s" % ("./plots/" + plot_name, games, training_episodes, learning_rate, "{0:.3g}".format(plotter.accuracies[-1])))


def evaluate_accuracy(samples, labels, value_function, silent=True):
    accuracy_sum = 0
    evaluation_samples = len(samples)//10
    for i in range(evaluation_samples):
        prediction = value_function.evaluate(samples[i])
        accuracy_sum += (prediction > (config.LABEL_WIN - config.LABEL_LOSS)/2) + config.LABEL_LOSS == labels[i]
        if not silent:
            print("Sample: %s, Label: %s, Prediction: %s" % (i, labels[i], "{0:.3g}".format(prediction)))
    return accuracy_sum/evaluation_samples


""" Configure Parameters here, adjust Network in valueFunction.SimpleValueFunction """

# test_with_parameters(games=300, training_episodes=150, learning_rate=float(round(0.1**3, 7)), plot_name="ReLU 2Layers")

for i in [3, 4, 3.5]:
    test_with_parameters(games=300, training_episodes=150, learning_rate=float(round(0.1**i, 7)), plot_name="ReLU 3FCLayers")
