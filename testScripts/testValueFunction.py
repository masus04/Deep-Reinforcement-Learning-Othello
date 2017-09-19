import config
from valueFunction import ValueFunction, SimpleValueFunction, FCValueFunction
from testScripts.generateDataSet import generate_greedy_data_set
from plotter import Plotter
from datetime import datetime


def test_with_parameters(games, training_episodes, plot_name):
    start_time = datetime.now()

    plotter = Plotter()

    samples, labels = generate_greedy_data_set(games, silent=True)
    print("Simulating %s games took %s" % (games, datetime.now()-start_time))
    start_time = datetime.now()

    value_function = SimpleValueFunction(plotter)
    for i in range(training_episodes):
        print("Training episode no. %s" % i)
        value_function.update(samples, labels)
        plotter.add_accuracy(evaluate_accuracy(samples, labels, value_function))

    print("Training %s episodes for %s games took %s" % (training_episodes, games ,datetime.now()-start_time))
    print("Final accuracy: %s" % plotter.accuracies[-1])
    plotter.plot("%s, %sGames, Accuracy: %s" % (plot_name, games, "{0:.3g}".format(plotter.accuracies[-1])))


def evaluate_accuracy(samples, labels, value_function):
    accuracy_sum = 0
    evaluation_samples = len(samples)//10
    for i in range(evaluation_samples):
        accuracy_sum += (value_function.evaluate(samples[i]) > (config.LABEL_WIN - config.LABEL_LOSS)/2) + config.LABEL_LOSS == labels[i]
    return accuracy_sum/evaluation_samples


""" Configure Parameters here, adjust Network in valueFunction.SimpleValueFunction """


# for game_episodes in [(10, 50), (20, 50), (30, 50), (50, 50), (100, 50), (1000, 10)]:
#    test_with_parameters(game_episodes[0],game_episodes[1], 2)

test_with_parameters(100, 10, "2Layers")
