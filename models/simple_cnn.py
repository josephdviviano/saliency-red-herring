import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, test_sample):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=5,
                stride=2,
                padding=0,
            ),
            nn.ReLU(),
        )
        # make the last shape dynamic
        test_out = self.conv1(test_sample.unsqueeze(0))
        test_out = (test_out.view(test_out.size(0), -1))
        self.out = nn.Linear(test_out.shape[1], 2)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x