import matplotlib.pyplot as plt


class Plotter:
    losses = []

    def add_loss(self, loss):
        self.losses.append(loss)

    def plot_losses(self):
        plt.plot(self.losses)
        plt.xlabel("Average loss per episode")
        plt.show()
