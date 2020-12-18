## Assignment Objective
---------- Reach 99.4% Accuracy on MNIST dataset using less than 20k params-------------
### Reached 99.46% Accuracy first at 5th epoch
### Consistently the Accuracy is reaching greater than 99.28% from 4th epoch
### Number of parameters in the model - 13,496
### Receptive Field - 32

----------------------------------------------------------------
#####        Layer (type)        #####       Output Shape    #####     Param    ##### Receptive Layer
================================================================
#####            Conv2d-1            [-1, 8, 28, 28]              72
#####       BatchNorm2d-2            [-1, 8, 28, 28]              16
#####            Conv2d-3           [-1, 16, 28, 28]           1,152
#####       BatchNorm2d-4           [-1, 16, 28, 28]              32
#####         MaxPool2d-5           [-1, 16, 14, 14]               0
#####            Conv2d-6           [-1, 16, 14, 14]           2,304
#####       BatchNorm2d-7           [-1, 16, 14, 14]              32
#####            Conv2d-8           [-1, 16, 14, 14]           2,304
#####       BatchNorm2d-9           [-1, 16, 14, 14]              32
#####        MaxPool2d-10             [-1, 16, 7, 7]               0
#####           Conv2d-11             [-1, 32, 5, 5]           4,608
#####      BatchNorm2d-12             [-1, 32, 5, 5]              64
#####           Conv2d-13             [-1, 10, 3, 3]           2,880

## LOGS ##
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
