import matplotlib.pyplot as plt
import pandas as pd


class Plotter:
    losses = []
    accuracies = []

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_accuracy(self, accuracy):
        self.accuracies.append(accuracy)

    def plot(self, plot_name):
        losses = pd.Series(self.losses, name="losses")
        accuracies = pd.Series(self.accuracies, name="accuracies")
        df = pd.DataFrame([losses, accuracies])
        df = df.transpose()
        # df.losses.plot()
        # df.accuracies.plot(secondary_y=True, style="g")
        df.plot(secondary_y=["accuracies"], title=plot_name, legend=True)
        plt.title = plot_name
        plt.xlabel = "Episodes"
        plt.savefig('%s.png' % plot_name)


def print_inplace(text):
    import sys
    sys.stdout.write("\r%s" % (text))
    sys.stdout.flush()
