import config
from valueFunction import ValueFunction
from testScripts.generateDataSet import generate_greedy_data_set
from plotter import Plotter
from datetime import datetime

start_time = datetime.now()

games = 10
training_episodes = 50
plotter = Plotter()

samples, labels = generate_greedy_data_set(games, silent=True)
print("Simulating %s games took %s" % (games, datetime.now()-start_time))
start_time = datetime.now()

valueFunction = ValueFunction(plotter)
for i in range(training_episodes):
    print("Training episode no. %s" % i)
    valueFunction.update(samples, labels)

print("Training %s episodes for %s games took %s" % (training_episodes, games ,datetime.now()-start_time))

accuracy_sum = 0
for i in range(len(samples)):
    accuracy_sum += (valueFunction.evaluate(samples[i]) > config.EMPTY) + config.LABEL_LOSS == labels[i]

print("Accuracy: %s" % (accuracy_sum/len(samples)))
plotter.plot_losses("testValueFunction")
