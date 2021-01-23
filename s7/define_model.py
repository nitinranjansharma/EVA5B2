

# depth wise convolutions
# dilated convolutions


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block - 32x32x3 -> 10
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=(3, 3), padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),

        )  # output_size = 30 ,Reseptive Field 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(
                3, 3), dilation=2, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32)

        )

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 14 ,Reseptive Field 10

        # CONVOLUTION BLOCK 2
        self.depthwise = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.pointwise = nn.Conv2d(32, 64, kernel_size=1)

        # self.convblock3 = nn.Sequential(
        #    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, bias=False),
        #    nn.ReLU(),
        #    nn.BatchNorm2d(64),

        # ) # output_size = 12 ,Reseptive Field 12

        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2)  # output_size = 6 ,Reseptive Field 24

        # CONVOLUTION BLOCK 3
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),

        )  # output_size = 4 ,Reseptive Field 26

        # TRANSITION BLOCK 3
        self.pool3 = nn.MaxPool2d(2, 2)  # output_size = 2 ,Reseptive Field 52

        # CONVOLUTION BLOCK 4
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),

        )  # output_size = 4 ,Reseptive Field 54

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        )  # output_size = 1, Reseptive Field 54

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10,
                      kernel_size=(1, 1), padding=0, bias=False),

            # nn.BatchNorm2d(10),
            # nn.ReLU(),

        )  # output_size = 1, Reseptive Field 54

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        #x = self.convblock3(x)
        x = self.pool2(x)
        x = self.convblock4(x)
        x = self.pool3(x)
        x = self.convblock5(x)

        x = self.gap(x)
        x = self.convblock6(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
