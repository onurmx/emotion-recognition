# OnsuNet implementation in PyTorch

import torch
import utils

class OnsuNetPyTorch(utils.ImageClassificationBase):
    def __init__(self, num_classes=1000):
        super(OnsuNetPyTorch, self).__init__()
        # block 1
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=512)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = torch.nn.Dropout(0.4)

        # block 2
        self.conv3 = torch.nn.Conv2d(in_channels=512, out_channels=384, kernel_size=3, padding=1)
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features=384)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = torch.nn.Dropout(0.4)

        # block 3
        self.conv4 = torch.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=3, padding=1)
        self.batchnorm3 = torch.nn.BatchNorm2d(num_features=192)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = torch.nn.Dropout(0.4)

        # block 4
        self.conv5 = torch.nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.batchnorm4 = torch.nn.BatchNorm2d(num_features=384)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = torch.nn.Dropout(0.4)

        # block 5
        self.fc1 = torch.nn.Linear(in_features=384 * 3 * 3, out_features=256)
        self.batchnorm5 = torch.nn.BatchNorm1d(num_features=256)
        self.dropout5 = torch.nn.Dropout(0.3)

        # output
        self.fc2 = torch.nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        # block 1
        out = torch.nn.functional.relu(self.conv1(x))
        out = torch.nn.functional.relu(self.conv2(out))
        out = self.batchnorm1(out)
        out = self.pool1(out)
        out = self.dropout1(out)

        # block 2
        out = torch.nn.functional.relu(self.conv3(out))
        out = self.batchnorm2(out)
        out = self.pool2(out)
        out = self.dropout2(out)

        # block 3
        out = torch.nn.functional.relu(self.conv4(out))
        out = self.batchnorm3(out)
        out = self.pool3(out)
        out = self.dropout3(out)

        # block 4
        out = torch.nn.functional.relu(self.conv5(out))
        out = self.batchnorm4(out)
        out = self.pool4(out)
        out = self.dropout4(out)

        # block 5
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.relu(self.fc1(out))
        out = self.batchnorm5(out)

        # output
        out = self.dropout5(out)
        out = self.fc2(out)

        return out

def pt_OnsuNet(num_classes):
    return OnsuNetPyTorch(num_classes)