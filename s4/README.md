## Assignment Objective
---------- Reach 99.4% Accuracy on MNIST dataset using less than 20k params-------------

### Solution file - EVA4 - Session 2_nrs.ipynb
### Reached 99.46% Accuracy first at 5th epoch
### Consistently the Accuracy is reaching greater than 99.28% from 4th epoch
### Number of parameters in the model - 13,496
### Receptive Field - 32

### Used a convolution block of Convolution 2D and Batch Norm followed by Max pool after 2 blocks
### Model Summary

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1,bias=False) #input -28x28 Output-26x26 RF-3
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1,bias=False) #input -26x26 Output-24x24 RF-5
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2) #input -16x16 Output-14x14 RF-10
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1,bias=False)
        self.batchnorm3 = nn.BatchNorm2d(16) #input -14x14 Output-12x12 RF-12
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1,bias=False) #input -12x12 Output-10x10 RF-14
        self.batchnorm4 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2) #input -10x10 Output-5x5 RF-28
        self.conv5 = nn.Conv2d(16, 32, 3,bias=False) #input -5x5 Output-3x3 RF-30
        self.batchnorm5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 10, 3,bias=False) #input -3x3 Output-1x1 RF-32
        
        

    def forward(self, x):
        x = self.pool1(F.relu(self.batchnorm2(self.conv2(self.batchnorm1(F.relu(self.conv1(x)))))))
        x = self.pool2(self.batchnorm4(F.relu(self.conv4(self.batchnorm3(F.relu(self.conv3(x)))))))
        x = F.relu(self.conv6(self.batchnorm5(F.relu(self.conv5(x)))))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, 10)
        return F.log_softmax(x)
```

```
--------------------------------------------------------------------------------------------------
#####        Layer (type)        ----       Output Shape    ----     Param    ---- Receptive Layer
==================================================================================================
#####            Conv2d-1  ------>          [-1, 8, 28, 28]   ------>          72  ------> 3
#####       BatchNorm2d-2  ------>          [-1, 8, 28, 28]   ------>           16 ------> 3
#####            Conv2d-3  ------>         [-1, 16, 28, 28]   ------>        1,152 ------> 5
#####       BatchNorm2d-4  ------>         [-1, 16, 28, 28]   ------>          32  ------> 5
#####         MaxPool2d-5  ------>         [-1, 16, 14, 14]   ------>            0 ------> 10
#####            Conv2d-6  ------>         [-1, 16, 14, 14]   ------>        2,304 ------> 12
#####       BatchNorm2d-7  ------>         [-1, 16, 14, 14]   ------>           32 ------> 12
#####            Conv2d-8  ------>         [-1, 16, 14, 14]   ------>        2,304 ------> 14
#####       BatchNorm2d-9  ------>         [-1, 16, 14, 14]   ------>           32 ------> 14
#####        MaxPool2d-10  ------>           [-1, 16, 7, 7]   ------>            0 ------> 28
#####           Conv2d-11  ------>           [-1, 32, 5, 5]   ------>        4,608 ------> 30
#####      BatchNorm2d-12  ------>           [-1, 32, 5, 5]   ------>           64 ------> 30
#####           Conv2d-13  ------>           [-1, 10, 3, 3]   ------>        2,880 ------> 32
================================================================
Total params: 13,496
Trainable params: 13,496
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.43
Params size (MB): 0.05
Estimated Total Size (MB): 0.48
----------------------------------------------------------------


```
## LOGS ##
```python
0%|          | 0/1875 [00:00<?, ?it/s]/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:26: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.018312253057956696 batch_id=1874: 100%|██████████| 1875/1875 [00:23<00:00, 80.22it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0430, Accuracy: 9882/10000 (99%)

loss=0.022196955978870392 batch_id=1874: 100%|██████████| 1875/1875 [00:23<00:00, 79.60it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0287, Accuracy: 9914/10000 (99%)

loss=0.008460606448352337 batch_id=1874: 100%|██████████| 1875/1875 [00:23<00:00, 79.08it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0320, Accuracy: 9898/10000 (99%)

loss=0.0011016001226380467 batch_id=1874: 100%|██████████| 1875/1875 [00:23<00:00, 80.23it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0235, Accuracy: 9931/10000 (99%)

loss=0.00027433180366642773 batch_id=1874: 100%|██████████| 1875/1875 [00:23<00:00, 78.56it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0171, Accuracy: 9946/10000 (99%)

loss=0.007730674464255571 batch_id=1874: 100%|██████████| 1875/1875 [00:23<00:00, 80.20it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0191, Accuracy: 9940/10000 (99%)

loss=0.005329648032784462 batch_id=1874: 100%|██████████| 1875/1875 [00:23<00:00, 79.14it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0206, Accuracy: 9928/10000 (99%)

loss=0.010548820719122887 batch_id=1874: 100%|██████████| 1875/1875 [00:23<00:00, 79.45it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0189, Accuracy: 9932/10000 (99%)

loss=0.0009434012463316321 batch_id=1874: 100%|██████████| 1875/1875 [00:23<00:00, 80.07it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0175, Accuracy: 9944/10000 (99%)

loss=0.002365692751482129 batch_id=1874: 100%|██████████| 1875/1875 [00:23<00:00, 79.61it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0209, Accuracy: 9941/10000 (99%)

loss=0.003850070759654045 batch_id=1874: 100%|██████████| 1875/1875 [00:23<00:00, 79.61it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0188, Accuracy: 9938/10000 (99%)

loss=0.0045530349016189575 batch_id=1874: 100%|██████████| 1875/1875 [00:23<00:00, 79.63it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0202, Accuracy: 9946/10000 (99%)

loss=9.08488582354039e-05 batch_id=1874: 100%|██████████| 1875/1875 [00:23<00:00, 80.87it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0191, Accuracy: 9940/10000 (99%)

loss=0.0014012942556291819 batch_id=1874: 100%|██████████| 1875/1875 [00:23<00:00, 80.58it/s]

Test set: Average loss: 0.0184, Accuracy: 9941/10000 (99%)
```
