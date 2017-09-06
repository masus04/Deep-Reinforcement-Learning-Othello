import torch
import torch.nn.functional as F
from torch.autograd import Variable


class ValueFunction:

    def __init__(self):
        self.model = Model()

    def evaluate(self, board):
        tensor = torch.FloatTensor([[board.board]])

        if torch.cuda.is_available():
            tensor = tensor.cuda(0)

        return self.model(Variable(tensor)).data[0][0]


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
        self.conv8 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=1, kernel_size=1, padding=0)
        self.fc1 = torch.nn.Linear(in_features=self.conv_to_linear_params_size, out_features=1)

        self.learning_rate = 0.01
        self.criterion = torch.nn.MSELoss(size_average=False)
        # self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = x.view(-1, self.conv_to_linear_params_size)
        return F.sigmoid(self.fc1(x)) # + config.LABEL_LOSS
