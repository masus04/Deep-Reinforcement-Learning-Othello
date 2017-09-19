import torch
import torch.nn.functional as F
from torch.autograd import Variable

import config


class ValueFunction:

    def __init__(self, plotter):
        self.plotter = plotter
        self.model = Model()
        self.learning_rate = config.LEARNING_RATE
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.CrossEntropyLoss()

    def evaluate(self, board_sample):
        tensor = torch.FloatTensor([[board_sample]])

        if torch.cuda.is_available():
            tensor = tensor.cuda(0)

        return self.model(Variable(tensor)).data[0][0][0][0]

    def update(self, training_samples, training_labels):
        minibatches_s = self.__generate_minibatches__(training_samples)
        minibatches_l = self.__generate_minibatches__(training_labels)

        accumulated_loss = 0
        for minibatch_samples, minibatch_labels in zip(minibatches_s, minibatches_l):
            if torch.cuda.is_available():
                minibatch_samples, minibatch_labels = minibatch_samples.cuda(0), minibatch_labels.cuda(0)

            self.optimizer.zero_grad()
            output = self.model(minibatch_samples)
            loss = self.criterion(output, minibatch_labels)
            loss.backward()
            self.optimizer.step()

            accumulated_loss += loss.data[0]

        # print("Average episode loss: %s for final label: %s" % (accumulated_loss/len(minibatches_s), minibatches_l[-1][-1].data[0]))
        self.plotter.add_loss(accumulated_loss/len(minibatches_s))

    @staticmethod
    def __generate_minibatches__(lst):
        return [Variable(torch.FloatTensor([lst[i:i+config.MINIBATCH_SIZE]])) for i in range(0, len(lst), config.MINIBATCH_SIZE)]


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv_channels = 8
        self.conv_to_linear_params_size = 1*8*8

        self.conv1 = torch.nn.Conv2d(in_channels= 1, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv7 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv8 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=1,                  kernel_size=1, padding=0)
        self.conv9 = torch.nn.Conv2d(in_channels=1,                  out_channels=1,                  kernel_size=8, padding=0)

        # self.fc1 = torch.nn.Linear(in_features=self.conv_to_linear_params_size, out_features=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        return F.sigmoid(self.conv9(x)) + config.LABEL_LOSS
        # x = x.view(-1, self.conv_to_linear_params_size)
        # return F.sigmoid(self.fc1(x)) + config.LABEL_LOSS


class SimpleValueFunction():

    def __init__(self, plotter):
        self.plotter = plotter
        self.model = SimpleModel()
        self.learning_rate = config.LEARNING_RATE
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.CrossEntropyLoss()

    def evaluate(self, board_sample):
        tensor = torch.FloatTensor([[board_sample]])

        if torch.cuda.is_available():
            tensor = tensor.cuda(0)

        result = self.model(Variable(tensor)).data[0]
        arr = [result[0], result[1]]
        return arr.index(max(result[0], result[1]))

    def update(self, training_samples, training_labels):
        minibatches_s = self.__generate_minibatches__(training_samples)
        minibatches_l = Variable(torch.LongTensor(training_labels))

        accumulated_loss = 0
        if torch.cuda.is_available():
            minibatches_s, minibatches_l = minibatches_s.cuda(0), minibatches_l.cuda(0)

        self.optimizer.zero_grad()
        output = self.model(minibatches_s)
        loss = self.criterion(output, minibatches_l)
        loss.backward()
        self.optimizer.step()

        accumulated_loss += loss.data[0]

        # print("Average episode loss: %s for final label: %s" % (accumulated_loss/len(minibatches_s), minibatches_l[-1][-1].data[0]))
        self.plotter.add_loss(accumulated_loss/len(minibatches_s))

    @staticmethod
    def __generate_minibatches__(lst, tensor=torch.FloatTensor):
        return Variable(tensor([[elem] for elem in lst]))


class SimpleModel(torch.nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()

        # Experiment 1
        # self.final = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=8, padding=0)

        # Experiment 2
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=9, padding=4)
        self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=9, padding=4)

        self.convToLinearFeatures = 8*8*4
        self.fc = torch.nn.Linear(in_features=self.convToLinearFeatures, out_features=2)

        # Experiment 3
        # self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
        # self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, padding=2)
        # self.conv3 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, padding=2)
        # self.conv4 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, padding=2)

        # Final Layer
        self.final = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=8, padding=0)

    def forward(self, x):

        x = F.sigmoid(self.conv1(x))
        x = F.sigmoid(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))

        x = x.view(-1, self.convToLinearFeatures)
        x = self.fc(x)
        return x

        # return F.sigmoid(self.final(x)) + config.LABEL_LOSS
