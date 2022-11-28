# Onsunet implementation in PyTorch

import torch

class Onsunet(torch.nn.Module):
    def __init__(self, num_classes=7):
        super(Onsunet, self).__init__()
        # block 1
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=192, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=192)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = torch.nn.Dropout(0.4)

        # block 2
        self.conv2 = torch.nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_features=256)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = torch.nn.Dropout(0.4)

        # block 3
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=320, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=320, out_channels=384, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(num_features=384)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = torch.nn.Dropout(0.4)

        # block 4
        self.conv5 = torch.nn.Conv2d(in_channels=384, out_channels=448, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=448, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(num_features=512)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = torch.nn.Dropout(0.4)

        # block 5
        self.fc1 = torch.nn.Linear(in_features=512*3*3, out_features=256)
        self.bn5 = torch.nn.BatchNorm1d(num_features=256)
        self.dropout5 = torch.nn.Dropout(0.3)

        # output
        self.fc2 = torch.nn.Linear(in_features=256, out_features=num_classes)
        
    def forward(self, x):
        # block 1
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # block 2
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # block 3
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # block 4
        x = torch.nn.functional.relu(self.conv5(x))
        x = torch.nn.functional.relu(self.conv6(x))
        x = self.bn4(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        # block 5
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.bn5(x)
        x = self.dropout5(x)

        # output
        x = self.fc2(x)
        return x