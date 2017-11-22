from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import src.config as config


class ValueFunction:

    def __init__(self, learning_rate=config.LEARNING_RATE):
        self.model = Model()
        if config.CUDA:
            self.model.cuda(0)
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

    def evaluate(self, board_sample):
        tensor = torch.FloatTensor([[board_sample]])

        if config.CUDA:
            tensor = tensor.cuda(0)

        return self.model(Variable(tensor)).data[0][0]

    def update(self, training_samples, training_labels):
        minibatches_s = self.__generate_minibatches__(training_samples)
        minibatches_l = self.__generate_minibatches__(training_labels)

        accumulated_loss = 0
        for minibatch_samples, minibatch_labels in zip(minibatches_s, minibatches_l):
            if config.CUDA:
                minibatch_samples, minibatch_labels = minibatch_samples.cuda(0), minibatch_labels.cuda(0)

            self.optimizer.zero_grad()
            output = self.model(minibatch_samples)
            loss = self.criterion(output, minibatch_labels)
            loss.backward()
            self.optimizer.step()

            accumulated_loss += abs(loss.data[0])

        # print("Average episode loss: %s for final label: %s" % (accumulated_loss/len(minibatches_s), minibatches_l[-1][-1].data[0]))
        return accumulated_loss/len(minibatches_s)

    @staticmethod
    def __generate_minibatches__(lst):
        return [Variable(torch.FloatTensor([lst[i:i+config.MINIBATCH_SIZE]])) for i in range(0, len(lst), config.MINIBATCH_SIZE)]

    def copy(self):
        value_function = self.__class__(learning_rate=self.learning_rate)
        value_function.model = deepcopy(self.model)
        value_function.optimizer = deepcopy(self.optimizer)
        return value_function


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv_channels = 8
        self.conv_to_linear_params_size = self.conv_channels*8*8

        self.conv1 = torch.nn.Conv2d(in_channels= 1, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv7 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(in_features=self.conv_to_linear_params_size, out_features=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = x.view(-1, self.conv_to_linear_params_size)
        return F.sigmoid(self.fc1(x)) + config.LABEL_LOSS


class SimpleValueFunction(ValueFunction):

    def __init__(self, learning_rate=config.LEARNING_RATE):
        super(SimpleValueFunction, self).__init__()
        self.model = SimpleModel()
        if config.CUDA:
            self.model.cuda(0)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)


class SimpleModel(torch.nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()

        self.conv_to_linear_params_size = 8*8*4

        # Experiment 1
        # self.final = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=8, padding=0)

        # Experiment 2
        # self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=9, padding=4)
        # self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=9, padding=4)

        # Experiment 3
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, padding=2)
        self.conv4 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, padding=2)

        # Final Layer
        # self.final = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=8, padding=0)
        self.final = torch.nn.Linear(in_features=self.conv_to_linear_params_size, out_features=1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, self.conv_to_linear_params_size)
        return F.sigmoid(self.final(x)) + config.LABEL_LOSS


class FCValueFunction(ValueFunction):

    def __init__(self, learning_rate=config.LEARNING_RATE):
        super(FCValueFunction, self).__init__()
        self.model = FCModel()
        if config.CUDA:
            self.model.cuda(0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)


class FCModel(torch.nn.Module):

    def __init__(self):
        super(FCModel, self).__init__()

        self.fc1 = torch.nn.Linear(in_features=64, out_features=64*10)
        self.fc2 = torch.nn.Linear(in_features=64*10, out_features=64*10)
        self.fc3 = torch.nn.Linear(in_features=64*10, out_features=1)

    def forward(self, x):

        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return F.sigmoid(x) + config.LABEL_LOSS


class NoValueFunction:

    def evaluate(self, board_sample):
        pass

    def update(self, training_samples, training_labels):
        pass

    def use_cuda(self, cuda):
        pass
