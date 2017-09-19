import matplotlib.pyplot as plt


class Plotter:
    losses = []

    def add_loss(self, loss):
        self.losses.append(loss)

    def plot_losses(self, plotName):
        plt.plot(self.losses)
        plt.ylabel("Average loss")
        plt.xlabel("Episode")
        plt.savefig('%s.png' % plotName);
        # plt.show()
