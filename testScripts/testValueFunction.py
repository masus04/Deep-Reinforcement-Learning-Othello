import config
from valueFunction import ValueFunction, SimpleValueFunction
from testScripts.generateDataSet import generate_greedy_data_set
from plotter import Plotter
from datetime import datetime


def test_with_parameters(games, training_episodes, layers_for_plotter):
    start_time = datetime.now()

    plotter = Plotter()

    samples, labels = generate_greedy_data_set(games, silent=True)
    print("Simulating %s games took %s" % (games, datetime.now()-start_time))
    start_time = datetime.now()

    valueFunction = SimpleValueFunction(plotter)
    for i in range(training_episodes):
        print("Training episode no. %s" % i)
        valueFunction.update(samples, labels)

    print("Training %s episodes for %s games took %s" % (training_episodes, games ,datetime.now()-start_time))

    accuracy_sum = 0
    for i in range(len(samples)):
        accuracy_sum += (valueFunction.evaluate(samples[i]) > config.EMPTY) + config.LABEL_LOSS == labels[i]

    print("Accuracy: %s" % '{0:.2f}'.format(accuracy_sum/len(samples)))
    plotter.plot_losses("%sLayers, %sGames" % (layers_for_plotter, games))

""" Configure Parameters here, adjust Network in valueFunction.SimpleValueFunction """

for game_episodes in [(10, 50), (20, 50), (30, 50), (50, 50), (100, 50), (1000, 10)]:
    test_with_parameters(game_episodes[0],game_episodes[1], 2)

# test_with_parameters(1000,10, 1)
