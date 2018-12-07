import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, test_sample):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(128),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(128),
            nn.ReLU())
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=6,
                stride=2,
                padding=0,
            ),
            nn.ReLU())
        
        self.pool = nn.MaxPool2d(20, stride=1)
        
        # make the last shape dynamic
        test_out = self.conv1(test_sample.unsqueeze(0))
        test_out = self.conv2(test_out)
        test_out = self.conv3(test_out)
        test_out = self.pool(test_out)
        
        test_out = (test_out.view(test_out.size(0), -1))
        
        self.out = nn.Sequential(
            nn.Linear(test_out.shape[1], 100),
            nn.ReLU(),
            nn.Linear(100, 2))

    def forward(self, x):
        res = {}
        res['conv1'] = self.conv1(x)
        res['conv2'] = self.conv2(res['conv1'])
        res['conv3'] = self.conv3(res['conv2'])
        res['pool'] = self.pool(res['conv3'])
        x = res['pool']
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, res
    
    
    