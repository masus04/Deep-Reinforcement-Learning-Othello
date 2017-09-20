import matplotlib.pyplot as plt
import pandas as pd


class Plotter:
    losses = []
    accuracies = []
    results = []

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_accuracy(self, accuracy):
        self.accuracies.append(accuracy)

    def add_result(self, result):
        self.results.append(result)

    def plot_accuracy(self, plot_name, resolution=False):
        """
        @plot_name: The name under which to save the plot
        @resolution: The number of points plotted. Losses and results will be averaged in groups of [resolution]"""
        return plot_two_lines("losses", self.losses, "accuracies", self.accuracies, plot_name, resolution)

    def plot_results(self, plot_name, resolution=False):
        """
        @plot_name: The name under which to save the plot
        @resolution: The number of points plotted. Losses and results will be averaged in groups of [resolution]"""
        return plot_two_lines("losses", self.losses, "results", self.results, plot_name, resolution)


def plot_two_lines(line1_name, line1_values, line2_name, line2_values, plot_name, resolution=False):
    """
    @plot_name: The name under which to save the plot
    @resolution: The number of points plotted. Losses and results will be averaged in groups of [resolution]"""
    line1 = pd.Series(chunk_list(line1_values, resolution) if resolution else line1_values, name=line1_name)
    line2 = pd.Series(chunk_list(line2_values, resolution) if resolution else line2_values, name=line2_name)
    df = pd.DataFrame([line1, line2])
    df = df.transpose()
    df.plot(secondary_y=[line2_name], title=plot_name, legend=True)
    plt.title = plot_name
    plt.xlabel = "Episodes"
    plt.savefig("./plots/%s.png" % plot_name)


def print_inplace(text, percentage, time_taken=None):
    import sys
    length_factor = 5
    progress_bar = int(round(percentage/length_factor)) * "*" + (round((100-percentage)/length_factor)) * "."
    progress_bar = progress_bar[:round(len(progress_bar)/2)] + "|" + str(round(percentage)) + "%|" + progress_bar[round(len(progress_bar)/2):]
    sys.stdout.write("\r%s |%s|" % (text, progress_bar) + (" Time: %s" % str(time_taken).split(".")[0] if time_taken else ""))
    sys.stdout.flush()


def chunk_list(lst, lst_size):
  if lst_size <= 1:
    return [sum(lst)/len(lst)]

  sublist = lst[0:len(lst)//lst_size]
  return [sum(sublist)/len(sublist)] + chunk_list(lst[len(lst)//lst_size:], lst_size-1)
