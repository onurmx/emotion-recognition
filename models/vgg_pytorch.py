# VGG-16 implementation in PyTorch

import torch

class VggTorch(torch.nn.Module):
    def __init__(self):
        super(VggTorch, self).__init__()

        # block 1
        self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.maxpool1 = torch.nn.MaxPool2d(2, 2)

        # block 2
        self.conv2_1 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.maxpool2 = torch.nn.MaxPool2d(2, 2)

        # block 3
        self.conv3_1 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.maxpool3 = torch.nn.MaxPool2d(2, 2)

        # block 4
        self.conv4_1 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.maxpool4 = torch.nn.MaxPool2d(2, 2)

        # block 5
        self.conv5_1 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.maxpool5 = torch.nn.MaxPool2d(2, 2)

        # fully connected layer
        self.fc14 = torch.nn.Linear(512 * 7 * 7, 4096)
        self.fc15 = torch.nn.Linear(4096, 4096)
        self.fc16 = torch.nn.Linear(4096, 1000)

    def forward(self, x):
        # block 1
        x = torch.nn.functional.relu(self.conv1_1(x))
        x = torch.nn.functional.relu(self.conv1_2(x))
        x = self.maxpool1(x)

        # block 2
        x = torch.nn.functional.relu(self.conv2_1(x))
        x = torch.nn.functional.relu(self.conv2_2(x))
        x = self.maxpool2(x)

        # block 3
        x = torch.nn.functional.relu(self.conv3_1(x))
        x = torch.nn.functional.relu(self.conv3_2(x))
        x = torch.nn.functional.relu(self.conv3_3(x))
        x = self.maxpool3(x)

        # block 4
        x = torch.nn.functional.relu(self.conv4_1(x))
        x = torch.nn.functional.relu(self.conv4_2(x))
        x = torch.nn.functional.relu(self.conv4_3(x))
        x = self.maxpool4(x)

        # block 5
        x = torch.nn.functional.relu(self.conv5_1(x))
        x = torch.nn.functional.relu(self.conv5_2(x))
        x = torch.nn.functional.relu(self.conv5_3(x))
        x = self.maxpool5(x)

        # fully connected layer
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc14(x))
        x = torch.nn.functional.relu(self.fc15(x))
        x = torch.nn.functional.softmax(self.fc16(x))

        return x