import sys
import os
import math
import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import pandas as pd


class Plotter:

    def __init__(self, player, plotter=None):
        self.player = player
        self.lr, self.alpha, self.e = (plotter.lr, plotter.alpha, plotter.e) if plotter else (player.value_function.learning_rate, player.alpha, player.e)
        self.plot_name = "%s lr:%s a:%s e:%s" % (player.player_name, self.lr, self.alpha, self.e)

        self.num_episodes = plotter.num_episodes if plotter else 0
        self.history_size = 1000

        self.losses = DataResolutionManager(plotter.losses if plotter else [], storage_size=self.history_size)
        self.accuracies = DataResolutionManager(plotter.accuracies if plotter else [], storage_size=self.history_size)
        self.results = DataResolutionManager(plotter.results if plotter else [], storage_size=self.history_size)
        self.evaluation_scores = DataResolutionManager(plotter.evaluation_scores if plotter else [], storage_size=self.history_size)
        self.last10Results = DataResolutionManager(plotter.last10Results if plotter else [], storage_size=self.history_size)

    def add_to_player_name(self, suffix):
        self.plot_name = self.plot_name.replace(self.player.player_name, self.player.player_name+suffix, 1)

    def add_loss(self, loss):
        self.losses.append(abs(loss))

    def add_accuracy(self, accuracy):
        self.num_episodes += 1
        self.accuracies.append(accuracy)

    def add_result(self, result):
        self.num_episodes += 1
        self.results.append(result)
        self.last10Results.append(sum(self.results.get_values()[-20:])/(20 if len(self.results.get_values()) > 20 else len(self.results.get_values())))

    def add_evaluation_score(self, score):
        self.evaluation_scores.append(score)

    def plot_accuracy(self, experiment_name="", comment="", path="/"):
        self.plot_two_lines("losses", self.losses.get_values(), "accuracies", self.accuracies.get_values(), "ACCURACIES " + self.create_file_name(experiment_name, comment), path=path)
        plt.close("all")

    def plot_results(self, experiment_name="", comment="", path="/"):
        self.plot_two_lines("losses", self.losses.get_values(), "results", self.last10Results.get_values(), "RESULTS " + self.create_file_name(experiment_name, comment), path=path)
        plt.close("all")

    def plot_scores(self, experiment_name="", comment="", path="/"):

        scores = self.evaluation_scores.get_values()

        if len(self.losses.get_values()) > len(scores):
            old_indices = np.arange(0, len(scores))
            new_length = len(self.losses.get_values())
            new_indices = np.linspace(0, len(scores) - 1, new_length)
            spl = UnivariateSpline(old_indices, scores, k=1, s=0)
            scores = spl(new_indices)

        self.plot_two_lines("losses", self.losses.get_values(), "evaluation score", scores, "SCORES " + self.create_file_name(experiment_name, comment), path=path)
        plt.close("all")

    def copy(self):
        return Plotter(player=self.player, plotter=self)

    def plot_two_lines(self, line1_name, line1_values, line2_name, line2_values, file_name=".", path="/"):
        try:
            if len(line1_values) == 0:
                if len(line2_values) == 0:
                    return
                else:
                    line = pd.Series(line2_values, name=line2_name)
                    df = pd.DataFrame([line])
                    df = df.transpose()
                    df.plot(legend=True, figsize=(16, 9))
            else:
                line1 = pd.Series(line1_values, name=line1_name)
                line2 = pd.Series(line2_values, name=line2_name)
                line3_name = line2_name + " average (last 100)"
                line3 = pd.Series([(sum(line2_values[(i-100) if i > 100 else 0 :i])/(100 if i > 100 else i)) for i in range(1, len(line2_values)+1)], name=line3_name)
                df = pd.DataFrame([line1, line2, line3])
                df = df.transpose()
                df.plot(secondary_y=[line2_name, line3_name], legend=True, figsize=(16, 9))

            plt.title(self.player.player_name + "%s Episodes\nStrategy: %s | Lr: %s | alpha: %s | epsilon: %s " % (self.num_episodes, self.player.value_function.__class__.__name__, self.lr, self.alpha, self.e))
            plt.xlabel = "Episodes"
            plt.savefig("./plots" + path + "%s.png" % file_name)
        except Exception as e:
            import traceback, sys
            print("| %s |" % ("-" * 50))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
            print("| %s |" % ("-" * 50))

    def clear_plots(self, experiment_name, comment=""):
        """Clears all .png files that match a certain @pattern"""
        try:
            folder = "./plots"
            for file_name in os.listdir(folder):
                file_name = os.path.join(folder, file_name)
                if os.path.isfile(file_name) and ".png" in file_name and self.plot_name in file_name and experiment_name in file_name and comment in file_name:
                    os.unlink(file_name)
        except Exception as e:
            print(e)

    def create_file_name(self, experiment_name, comment=""):
        return "%s Episodes:%s %s %s" % (self.plot_name, self.num_episodes, experiment_name, comment)


class DataResolutionManager:

    def __init__(self, data_points=[], storage_size=1000):
        try:  # data_points can be either DataResolutionManagers or simple lists
            data_points = data_points.get_values()
        except AttributeError:
            pass

        data_points = data_points.copy()
        self.storage_size = storage_size
        self.compression_factor = math.ceil((len(data_points)+1) / storage_size)
        self.values = []
        self.buffer = []

        if self.compression_factor > 1:
            self.buffer = data_points[len(data_points) - len(data_points) % self.compression_factor:]

            for i in range(len(data_points) // self.compression_factor):
                self.values.append(sum(data_points[:self.compression_factor]) / self.compression_factor)
                data_points = data_points[self.compression_factor:]
        else:
            self.values = data_points

    def append(self, value):
        self.buffer.append(value)
        if len(self.buffer) >= self.compression_factor:
            self.values.append(sum(self.buffer) / len(self.buffer))
            self.buffer = []
            if len(self.values) >= 2*self.storage_size:
                if len(self.values) % 2 != 0:  # Move uneven element back to buffer
                    self.buffer.append(self.values.pop())
                self.values = [(a + b) / 2 for a, b in zip(self.values[0::2], self.values[1::2])]
                self.compression_factor *= 2

    def get_values(self):
        if len(self.buffer) == 0:
            return self.values
        else:
            return self.values + [sum(self.buffer) / len(self.buffer)]


""" DEPRECATED?
def chunk_list(lst, lst_size):
  if lst_size <= 1:
    return [sum(lst)/len(lst)]

  sublist = lst[0:len(lst)//lst_size]
  return [sum(sublist)/len(sublist)] + chunk_list(lst[len(lst)//lst_size:], lst_size-1)
"""


class NoPlotter:

    def add_loss(self, loss):
        pass

    def add_accuracy(self, accuracy):
        pass

    def add_result(self, result):
        pass

    def add_evaluation_score(self, score):
        pass

    def plot_accuracy(self, comment=""):
        pass

    def plot_results(self, comment=""):
        pass

    def plot_scores(self, comment=""):
        pass


class Printer:

    @staticmethod
    def print_inplace(text, percentage, time_taken=None, comment=""):
        percentage = int(percentage)
        length_factor = 5
        progress_bar = int(round(percentage/length_factor)) * "*" + (round((100-percentage)/length_factor)) * "."
        progress_bar = progress_bar[:round(len(progress_bar)/2)] + "|" + str(int(percentage)) + "%|" + progress_bar[round(len(progress_bar)/2):]
        sys.stdout.write("\r%s |%s|" % (text, progress_bar) + (" Time: %s" % str(time_taken).split(".")[0] if time_taken else "") + comment)
        sys.stdout.flush()

        if percentage == 100:
            print()
