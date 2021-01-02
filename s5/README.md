

1. Target - Reach 99.4% accuracy in Validation within 15epochs using 10k params
2. Experimented on regularizer, augmentation, learning rate, and scheduler to reach the desired result
3. Experimented with Different architectures
4. Files are in order of experiments - EVA4S5F9_experiment1.ipynb < EVA4S5F9_experiment2.ipynb < EVA4S5F9_experiment3.ipynb
5. EVA4S5F9_Final.ipynb is the final notebook that can be used for evaluation

We tried to experiment with bunch of model. But at end we tried different iteration of Same model with different learning rate & Schedular & Created trimmed version of the those model to get the final model

### Iteration 1 

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
              ReLU-2           [-1, 10, 26, 26]               0
       BatchNorm2d-3           [-1, 10, 26, 26]              20
           Dropout-4           [-1, 10, 26, 26]               0
            Conv2d-5           [-1, 10, 24, 24]             900
              ReLU-6           [-1, 10, 24, 24]               0
       BatchNorm2d-7           [-1, 10, 24, 24]              20
           Dropout-8           [-1, 10, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             100
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 16, 10, 10]           1,440
             ReLU-12           [-1, 16, 10, 10]               0
      BatchNorm2d-13           [-1, 16, 10, 10]              32
          Dropout-14           [-1, 16, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           2,304
             ReLU-16             [-1, 16, 8, 8]               0
      BatchNorm2d-17             [-1, 16, 8, 8]              32
          Dropout-18             [-1, 16, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           2,304
             ReLU-20             [-1, 16, 6, 6]               0
      BatchNorm2d-21             [-1, 16, 6, 6]              32
          Dropout-22             [-1, 16, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           2,304
             ReLU-24             [-1, 16, 6, 6]               0
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
================================================================
Total params: 9,770
Trainable params: 9,770
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.55
Params size (MB): 0.04
Estimated Total Size (MB): 0.59
```

It reached 99.4% but was not consistent enough, so we we tried the same model with different learning rate,below is the final two epoch

```python
EPOCH: 13
Loss=0.022484928369522095 Batch_id=937 Accuracy=99.10: 100%|██████████| 938/938 [01:42<00:00,  9.15it/s]
Loss=0.0023298657033592463 Batch_id=0 Accuracy=100.00:   0%|          | 1/938 [00:00<01:41,  9.25it/s]
Test set: Average loss: 0.0190, Accuracy: 9941/10000 (99.41%)

EPOCH: 14
Loss=0.014951080083847046 Batch_id=937 Accuracy=99.09: 100%|██████████| 938/938 [01:42<00:00,  9.18it/s]
Test set: Average loss: 0.0185, Accuracy: 9937/10000 (99.37%)
```

### Iteration 2

Same model with below change

```python
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
scheduler = StepLR(optimizer, step_size=10, gamma=0.01)
```
it reached the 99.4% bit more consistent but still had a random shift

```python
EPOCH: 12
Loss=0.01769702322781086 Batch_id=468 Accuracy=99.20: 100%|██████████| 469/469 [00:36<00:00, 12.86it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0198, Accuracy: 9942/10000 (99.42%)

EPOCH: 13
Loss=0.015365933068096638 Batch_id=468 Accuracy=99.17: 100%|██████████| 469/469 [00:36<00:00, 12.76it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0189, Accuracy: 9939/10000 (99.39%)

EPOCH: 14
Loss=0.017201708629727364 Batch_id=468 Accuracy=99.19: 100%|██████████| 469/469 [00:37<00:00, 12.57it/s]
Test set: Average loss: 0.0190, Accuracy: 9941/10000 (99.41%)
```

### Iteration 3

Change the scheduler a bit
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
```
For final few epoch it performed closely with the 2nd one

```python
EPOCH: 12
Loss=0.03713931515812874 Batch_id=468 Accuracy=99.10: 100%|██████████| 469/469 [00:42<00:00, 11.04it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0175, Accuracy: 9944/10000 (99.44%)

EPOCH: 13
Loss=0.021043211221694946 Batch_id=468 Accuracy=99.13: 100%|██████████| 469/469 [00:41<00:00, 11.39it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0176, Accuracy: 9940/10000 (99.40%)

EPOCH: 14
Loss=0.05626573786139488 Batch_id=468 Accuracy=99.11: 100%|██████████| 469/469 [00:40<00:00, 11.72it/s]
Test set: Average loss: 0.0178, Accuracy: 9937/10000 (99.37%)
```

### Final Model

For final model, model summary is given below

```python

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
              ReLU-2           [-1, 10, 26, 26]               0
       BatchNorm2d-3           [-1, 10, 26, 26]              20
            Conv2d-4           [-1, 16, 24, 24]           1,440
              ReLU-5           [-1, 16, 24, 24]               0
       BatchNorm2d-6           [-1, 16, 24, 24]              32
            Conv2d-7           [-1, 10, 24, 24]             160
       BatchNorm2d-8           [-1, 10, 24, 24]              20
              ReLU-9           [-1, 10, 24, 24]               0
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 10, 10, 10]             900
             ReLU-12           [-1, 10, 10, 10]               0
      BatchNorm2d-13           [-1, 10, 10, 10]              20
           Conv2d-14           [-1, 32, 10, 10]             320
      BatchNorm2d-15           [-1, 32, 10, 10]              64
             ReLU-16           [-1, 32, 10, 10]               0
           Conv2d-17           [-1, 10, 10, 10]             320
             ReLU-18           [-1, 10, 10, 10]               0
      BatchNorm2d-19           [-1, 10, 10, 10]              20
           Conv2d-20             [-1, 10, 8, 8]             900
             ReLU-21             [-1, 10, 8, 8]               0
      BatchNorm2d-22             [-1, 10, 8, 8]              20
           Conv2d-23             [-1, 32, 8, 8]             320
      BatchNorm2d-24             [-1, 32, 8, 8]              64
             ReLU-25             [-1, 32, 8, 8]               0
           Conv2d-26             [-1, 10, 8, 8]             320
             ReLU-27             [-1, 10, 8, 8]               0
      BatchNorm2d-28             [-1, 10, 8, 8]              20
           Conv2d-29             [-1, 14, 6, 6]           1,260
             ReLU-30             [-1, 14, 6, 6]               0
      BatchNorm2d-31             [-1, 14, 6, 6]              28
           Conv2d-32             [-1, 16, 4, 4]           2,016
             ReLU-33             [-1, 16, 4, 4]               0
      BatchNorm2d-34             [-1, 16, 4, 4]              32
        AvgPool2d-35             [-1, 16, 1, 1]               0
           Conv2d-36             [-1, 10, 1, 1]             160
================================================================
Total params: 8,546
Trainable params: 8,546
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.72
Params size (MB): 0.03
Estimated Total Size (MB): 0.76
----------------------------------------------------------------
```
below is the last few epochs 


Test set: Average loss: 0.0221, Accuracy: 9935/10000 (99.35%)

```python
EPOCH: 11
Loss=0.017562782391905785 Batch_id=468 Accuracy=99.21: 100%|██████████| 469/469 [00:43<00:00, 10.88it/s]
current Learing Rate:  0.0025
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0202, Accuracy: 9940/10000 (99.40%)

EPOCH: 12
Loss=0.00921888928860426 Batch_id=468 Accuracy=99.25: 100%|██████████| 469/469 [00:42<00:00, 11.04it/s]
current Learing Rate:  0.0025
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0191, Accuracy: 9941/10000 (99.41%)

EPOCH: 13
Loss=0.06074872240424156 Batch_id=468 Accuracy=99.34: 100%|██████████| 469/469 [00:43<00:00, 10.88it/s]
current Learing Rate:  0.0025
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0199, Accuracy: 9937/10000 (99.37%)

EPOCH: 14
Loss=0.025518519803881645 Batch_id=468 Accuracy=99.31: 100%|██████████| 469/469 [00:43<00:00, 10.73it/s]
current Learing Rate:  0.0025
Test set: Average loss: 0.0182, Accuracy: 9940/10000 (99.40%)

```

### Factor deciding the final model
1. less parameters
2. Bit more consistent 

### Team
1. Rahul
2. Nilanjan
3. Nitin

