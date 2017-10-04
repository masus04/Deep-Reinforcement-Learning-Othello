import sys
import torch
if torch.cuda.is_available():
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


class Plotter:

    def __init__(self, plot_name):
        self.plot_name = plot_name
        self.losses = []
        self.accuracies = []
        self.results = []
        self.last10Results = []

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_accuracy(self, accuracy):
        self.accuracies.append(accuracy)

    def add_result(self, result):
        self.results.append(result)
        self.last10Results.append(sum(self.results[-20:])/(20 if len(self.results)>20 else len(self.results)))

    def plot_accuracy(self, plot_name, resolution=False):
        """
        @plot_name: The name under which to save the plot
        @resolution: The number of points plotted. Losses and results will be averaged in groups of [resolution]"""
        return self.plot_two_lines("losses", self.losses, "accuracies", self.accuracies, "%s, %s Episodes" % (self.plot_name, len(self.results)), resolution)

    def plot_results(self, resolution=False, comment=""):
        """
        @plot_name: The name under which to save the plot
        @resolution: The number of points plotted. Losses and results will be averaged in groups of [resolution]"""
        self.plot_two_lines("losses", self.losses, "results", self.results, "%s, %s Episodes%s" % (self.plot_name, len(self.results), comment), resolution)
        self.plot_two_lines("losses", self.losses, "results", self.last10Results, "HighRes - %s, %s Episodes%s" % (self.plot_name, len(self.results), comment), resolution=False)
        plt.close("all")

    @staticmethod
    def plot_two_lines(line1_name, line1_values, line2_name, line2_values, plot_name, resolution):
        """
        @plot_name: The name under which to save the plot
        @resolution: The number of points plotted. Losses and results will be averaged in groups of [resolution]"""
        line1 = pd.Series(chunk_list(line1_values, resolution) if resolution and resolution <= len(line1_values) else line1_values, name=line1_name)
        line2 = pd.Series(chunk_list(line2_values, resolution) if resolution and resolution <= len(line2_values) else line2_values, name=line2_name)
        df = pd.DataFrame([line1, line2])
        df = df.transpose()
        df.plot(secondary_y=[line2_name], title=plot_name, legend=True, figsize=(16, 9))
        plt.title = plot_name
        plt.xlabel = "Episodes"
        plt.savefig("./plots/%s.png" % plot_name)

    @staticmethod
    def print_inplace(text, percentage, time_taken=None):
        length_factor = 5
        progress_bar = int(round(percentage/length_factor)) * "*" + (round((100-percentage)/length_factor)) * "."
        progress_bar = progress_bar[:round(len(progress_bar)/2)] + "|" + str(int(percentage)) + "%|" + progress_bar[round(len(progress_bar)/2):]
        sys.stdout.write("\r%s |%s|" % (text, progress_bar) + (" Time: %s" % str(time_taken).split(".")[0] if time_taken else ""))
        sys.stdout.flush()


def chunk_list(lst, lst_size):
  if lst_size <= 1:
    return [sum(lst)/len(lst)]

  sublist = lst[0:len(lst)//lst_size]
  return [sum(sublist)/len(sublist)] + chunk_list(lst[len(lst)//lst_size:], lst_size-1)


class NoPlotter:

    def add_loss(self, loss):
        pass

    def add_accuracy(self, accuracy):
        pass

    def add_result(self, result):
        pass

    def plot_accuracy(self, resolution=False):
        pass

    def plot_results(self, resolution=False):
        pass


class Printer:

    def __init__(self):
        self.percentage = -1

    def print_inplace(self, text, percentage, time_taken=None):
        percentage = int(percentage)
        if percentage > self.percentage:
            self.percentage = percentage
            length_factor = 5
            progress_bar = int(round(percentage/length_factor)) * "*" + (round((100-percentage)/length_factor)) * "."
            progress_bar = progress_bar[:round(len(progress_bar)/2)] + "|" + str(int(percentage)) + "%|" + progress_bar[round(len(progress_bar)/2):]
            sys.stdout.write("\r%s |%s|" % (text, progress_bar) + (" Time: %s" % str(time_taken).split(".")[0] if time_taken else ""))
            sys.stdout.flush()

            if percentage == 100:
                self.reset()

    def reset(self):
        self.percentage = 0
        print()
