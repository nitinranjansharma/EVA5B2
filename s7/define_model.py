## new approach
import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32)
            
        ) #i/p=32, output_size = 30  Rf 3 Jout - 1

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, bias=False),  
            nn.ReLU(),
            nn.BatchNorm2d(64)
            
        ) # i/p=30,output_size = 32 RF 5 Jout -1

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),    
            
        ) # i/p=30,output_size = 32 RF 5 Jout -1

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # i./p=32,output_size = 16  RF 6 Jout - 2


        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False,dilation=2),  ##dilation
            nn.ReLU(),
            nn.BatchNorm2d(128)
           ) #i/p=16 output_size =12  RF 14 Jout - 2

       # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2) #i/p=12 output_size = 6  RF 16 Jout - 4
        
          #CONVOLUTION BLOCK 3
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128)
            ) # i/p=6,output_size =8  RF -24 ,Jout - 4

        # TRANSITION BLOCK 3
        self.pool3 = nn.MaxPool2d(2, 2) # i/p=8,output_size = 4  RF 32 Jout - 8

       #CONVOLUTION BLOCK 4

        self.depthwise = nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128)
        self.pointwise = nn.Conv2d(128, 64, kernel_size=1)

        # i/p=4,output_size =4  RF 48 Jout - 8


        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            
        ) 


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.convblock5(x)
        x = self.pool3(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.gap(x)        
        x = self.convblock9(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)