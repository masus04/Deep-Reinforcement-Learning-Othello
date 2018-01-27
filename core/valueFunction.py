import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
try:
    from torch.distributions import Categorical
except:
    from torch.distributions import Multinomial as Categorical
from torch.autograd import Variable

import core.config as config


""" | ---------- Models ---------- | """


class Model(torch.nn.Module):

    def __init__(self, decoupled=False, policy_gradient=False):
        super(Model, self).__init__()

        self.input_channels = 3 if decoupled else 1
        self.output_features = 64 if policy_gradient else 1
        self.policy_gradient = policy_gradient

        self.conv_channels = 8
        self.conv_to_linear_params_size = self.conv_channels*8*8

        self.conv1 = torch.nn.Conv2d(in_channels=self.input_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv7 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)

        self.fc1 = torch.nn.Linear(in_features=self.conv_to_linear_params_size, out_features=1)

        self.final_conv = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x, legal_moves_map=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = x.view(-1, self.conv_to_linear_params_size)
        x = self.fc1(x)

        if self.policy_gradient:
            return self.legal_moves_for_policy_gradient(x, legal_moves_map)

        else:
            return F.sigmoid(x) + config.LABEL_LOSS

    def legal_moves_for_policy_gradient(self, softmaxed_x, legal_moves_map):
        x = F.softmax(softmaxed_x, dim=1)

        # Set illegal move probabilities to 0 and regularize so that they sum up to 1
        x = torch.mul(x, legal_moves_map)
        x = x * 1 / x.sum()
        return x


class LargeModel(Model):
    def __init__(self, decoupled=False, policy_gradient=False):
        super(LargeModel, self).__init__()

        self.input_channels = 3 if decoupled else 1
        self.output_features = 64 if policy_gradient else 1
        self.policy_gradient = policy_gradient

        self.conv_channels = 16
        self.conv_to_linear_params_size = self.conv_channels*8*8

        self.conv1 = torch.nn.Conv2d(in_channels=self.input_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv7 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)

        self.fc1 = torch.nn.Linear(in_features=self.conv_to_linear_params_size, out_features=self.conv_to_linear_params_size//2)
        self.fc2 = torch.nn.Linear(in_features=self.conv_to_linear_params_size//2, out_features=self.output_features)

    def forward(self, x, legal_moves_map=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = x.view(-1, self.conv_to_linear_params_size)
        x = F.relu((self.fc1(x)))
        x = self.fc2(x)

        if self.policy_gradient:
            return self.legal_moves_for_policy_gradient(x, legal_moves_map)

        else:
            return F.sigmoid(x) + config.LABEL_LOSS


class HugeModel(Model):
    def __init__(self, decoupled=False, policy_gradient=False):
        super(HugeModel, self).__init__()

        self.input_channels = 3 if decoupled else 1
        self.output_features = 64 if policy_gradient else 1
        self.policy_gradient = policy_gradient
        
        self.conv_channels = 32
        self.conv_to_linear_params_size = self.conv_channels*8*8

        self.conv1 = torch.nn.Conv2d(in_channels=self.input_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv7 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)

        self.fc1 = torch.nn.Linear(in_features=self.conv_to_linear_params_size, out_features=self.conv_to_linear_params_size//2)
        self.fc2 = torch.nn.Linear(in_features=self.conv_to_linear_params_size//2, out_features=self.conv_to_linear_params_size//4)
        self.fc3 = torch.nn.Linear(in_features=self.conv_to_linear_params_size//4, out_features=self.conv_to_linear_params_size//8)
        self.fc4 = torch.nn.Linear(in_features=self.conv_to_linear_params_size//8, out_features=self.output_features)

    def forward(self, x, legal_moves_map=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = x.view(-1, self.conv_to_linear_params_size)
        x = F.relu((self.fc1(x)))
        x = F.relu((self.fc2(x)))
        x = F.relu((self.fc3(x)))
        x = self.fc4(x)

        if self.policy_gradient:
            return self.legal_moves_for_policy_gradient(x, legal_moves_map)

        else:
            return F.sigmoid(x) + config.LABEL_LOSS


class SimpleModel(torch.nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()

        self.conv_channels = 8
        self.conv_to_linear_params_size = self.conv_channels*8*8

        # Experiment 1
        # self.final = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=8, padding=0)

        # Experiment 2
        # self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=9, padding=4)
        # self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=9, padding=4)

        # Experiment 3
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=self.conv_channels, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=5, padding=2)
        self.conv4 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=5, padding=2)

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


""" | ---------- ValueFunctions ---------- | """


class ValueFunction:

    def __init__(self, learning_rate=config.LEARNING_RATE, model=Model()):
        self.model = model
        if config.CUDA:
            self.model.cuda(0)
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

    def data_reshape(self, board_sample):
        return [board_sample]

    def evaluate(self, board_sample):
        tensor = torch.FloatTensor([self.data_reshape(board_sample)])

        if config.CUDA:
            tensor = tensor.cuda(0)

        return self.model(Variable(tensor)).data[0][0]

    def update(self, training_samples, training_labels):
        training_samples = [self.data_reshape(sample) for sample in training_samples]
        minibatches_s = self.__generate_minibatches__(training_samples)
        minibatches_l = self.__generate_minibatches__(training_labels)

        accumulated_loss = 0
        for minibatch_samples, minibatch_labels in zip(minibatches_s, minibatches_l):
            if config.CUDA:
                minibatch_samples, minibatch_labels = minibatch_samples.cuda(0), minibatch_labels.cuda(0)

            # Normalization from PyTorch examples
            # minibatch_labels = (minibatch_labels - minibatch_labels.mean()) / (minibatch_labels.std() + np.finfo(np.float64).eps)

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
        return [Variable(torch.FloatTensor(lst[i:i+config.MINIBATCH_SIZE])) for i in range(0, len(lst), config.MINIBATCH_SIZE)]

    def copy(self):
        value_function = self.__class__(learning_rate=self.learning_rate, model=deepcopy(self.model))
        value_function.optimizer = deepcopy(self.optimizer)
        return value_function


class LargeValueFunction(ValueFunction):
    def __init__(self, learning_rate=config.LEARNING_RATE, model=LargeModel()):
        super(LargeValueFunction, self).__init__(learning_rate=learning_rate, model=model)


class HugeValueFunction(ValueFunction):
    def __init__(self, learning_rate=config.LEARNING_RATE, model=HugeModel()):
        super(HugeValueFunction, self).__init__(learning_rate=learning_rate, model=model)


class SimpleValueFunction(ValueFunction):
    def __init__(self, learning_rate=config.LEARNING_RATE, model=SimpleModel()):
        super(SimpleValueFunction, self).__init__(learning_rate=learning_rate, model=model)


""" | ---------- Decoupled Value Functions ---------- | """


class DecoupledValueFunction(ValueFunction):
    def __init__(self, learning_rate=config.LEARNING_RATE, model=Model(decoupled=True)):
        super(DecoupledValueFunction, self).__init__(learning_rate=learning_rate, model=model)

    def data_reshape(self, board_sample):
        black_board = board_sample == config.BLACK
        white_board = board_sample == config.WHITE
        empty_board = board_sample == config.EMPTY

        return np.array([black_board, white_board, empty_board], dtype=np.float)


class LargeDecoupledValueFunction(DecoupledValueFunction):
    def __init__(self, learning_rate=config.LEARNING_RATE, model=LargeModel(decoupled=True)):
        super(LargeDecoupledValueFunction, self).__init__(learning_rate=learning_rate, model=model)


class HugeDecoupledValueFunction(DecoupledValueFunction):
    def __init__(self, learning_rate=config.LEARNING_RATE, model=HugeModel(decoupled=True)):
        super(HugeDecoupledValueFunction, self).__init__(learning_rate=learning_rate, model=model)


""" | ---------- Policy Gradient VFs ---------- | """


class PGValueFunction(ValueFunction):
    def __init__(self, learning_rate=config.LEARNING_RATE, model=LargeModel(policy_gradient=True)):
        super(PGValueFunction, self).__init__(learning_rate=learning_rate, model=model)
        self.log_probs = []

    def evaluate(self, board_sample, legal_moves_map):
        input = Variable(torch.FloatTensor([self.data_reshape(board_sample)]))
        legal_moves_map = Variable(torch.FloatTensor(legal_moves_map)).view(-1, 64)
        probs = self.model(input, legal_moves_map)
        distribution = Categorical(probs)
        action = distribution.sample()
        move = (action.data[0] // 8, action.data[0] % 8)
        log_prob = distribution.log_prob(action)
        return move, log_prob

    def data_reshape(self, board_sample):
        return [board_sample]

    def update(self, log_probs, training_labels):
        label_batches = Variable(torch.FloatTensor(self.data_reshape(training_labels)))

        self.optimizer.zero_grad()
        policy_loss = [-log_prob*label for log_prob, label in zip(log_probs, label_batches)]
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.data[0]


class PGLargeValueFunction(PGValueFunction):
    def __init__(self, learning_rate=config.LEARNING_RATE, model=LargeModel(policy_gradient=True)):
        super(PGValueFunction, self).__init__(learning_rate=learning_rate, model=model)


""" | ---------- Special cases ---------- | """


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


class FCValueFunction(ValueFunction):
    def __init__(self, learning_rate=config.LEARNING_RATE):
        super(FCValueFunction, self).__init__()
        self.model = FCModel()
        if config.CUDA:
            self.model.cuda(0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)


class NoValueFunction:

    learning_rate = 0

    def evaluate(self, board_sample):
        pass

    def update(self, training_samples, training_labels):
        pass

    def use_cuda(self, cuda):
        pass
