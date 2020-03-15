import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, base=4):
        self.base = base
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 2**self.base, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2**self.base, 2**self.base, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(2**self.base, 2**(self.base+1), kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(2**(self.base+1), 2**(self.base+1), kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(2**(self.base+1), 2**(self.base+2), kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(2**(self.base+2), 2**(self.base+2), kernel_size=3, padding=1)

        self.conv7 = nn.Conv2d(2**(self.base+2), 2**(self.base+3), kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(2**(self.base+3), 2**(self.base+3), kernel_size=3, padding=1)

        self.conv9 = nn.Conv2d(2**(self.base+3), 2**(self.base+4), kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(2**(self.base+4), 2**(self.base+4), kernel_size=3, padding=1)

        self.pool = nn.AdaptiveMaxPool2d(kernel_size=2)

        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 1)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):
        # First conv block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Second conv block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        # Third conv block
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)

        # Fourth conv block
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv8(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense layers
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.linear3(x))
        return x
