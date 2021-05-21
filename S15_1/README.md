
### Model
#### Objective – Requirement suggested that we have to unify three different vision tasks under one model architecture:
* a.	Using YOLO architecture learn to identify specific objects(Hardhat, Vest, Mask, Boots)
* b.	Using MiDaS architecture learn to generate depth map on custom images
* c.	Using PlaneRCNN architecture learn to generate plane maps on custom images
The concepts of transfer learning being used to tune the weights based on the custom data created from sample images. 

#### MiDaS – reference: https://github.com/intel-isl/MiDaS
 MiDaS architecture is used as a backbone for the entire trio model architecture. The ResNext101 part of the model is used as is as encoder structure to learn on the image features maps. The MiDaS part serving as an encoder generating the required feature maps which were passed onto the decoder part formed by MiDaS(for MiDaS) , YOLOv3(for YOLOv3) and PlaneRCNN. Gradient flow was frozen in the first half of the architecture and no additional training was done on the same.
#### Model YOLO – reference: https://github.com/ultralytics/yolov3
YOLOv3 architecture was updated based on the already feature map extracted from the MiDaS fixed(encoder architecture). A header was generated to map the already created features from ResNext101 to match YOLOv3 decoder requirements followed by decoder YOLO part. 

#### PlaneRCNN – reference: https://github.com/NVlabs/planercnn
Initial purpose of PlaneRCNN was to generates planes from an image, detects arbitrary number of planes, and reconstructs piecewise planar surfaces from a single RGB image. The feature maps from the ResNext101 is followed into PlaneRCNN block matching up with the encoder output shapes to input of the decoder (PlaneRCNN). The feature pyramid network was used to capture the feature maps and train on the portion of MaskRCNN.

base: ![img](<basic_struct.PNG>)

### THE DATASET
As we have already gone through the model, which is a combination of MiDaS, YOLO and PlaneRCNN we have created our own dataset from all three submodels - MiDAS, YOLOv3 and PlaneRCNN, which further combined to create the master data. 
Manually collected and annotated data of ~3500 images was used as the initial start point. 

#### Create MiDaS Dataset
Original MiDaS github repo was used to create the MiDaS dataset ground truth. We ran our data on the pretrained MiDaS weights and generated the depth maps of all the input images.
 
#### Create YOLOv3 Dataset
In the case of YOLO we use the annotated labels created during previous assignments. In time to training we have used pretrained YOLOv3 weights.

#### Create PlaneRCNN Dataset
In the case of PlaneRCNN, we are using segmentation and masks output after validating on pretrained weight. We are not using the depth output as MiDaS does better depth predictions. The masks and segmentations of the output then combined to get a 2D image. 

#### Folder Structure Creation: 
folder: ![img](<folder_struct.PNG>)

### TRAINING
Training is majorly done using transfer learning strategy. The reason being, the lack of image and processing power and this process provide satisfactory results in limited time. 
 
#### MiDaS training
MiDaS was used as the backbone for training and the weights were frozen, only the loss calculation for MiDaS was implemented. Simple RMSE Loss was chosen.
 
In the dataset, the input images were letterboxed, so before applying the loss function, the letterbox paddings were removed (using the padding extents that were returned by the dataset), output was resized to match the ground truth size and then the RMSE loss was applied.
PlaneRCNN training
In the case of PlaneRCNN, issues encountered here were mostly related to depth and GPU vs CPU, but that was solved by disabling depth prediction in the PlaneRCNN config. Also did some code change to maintain and transfer the data to the correct device. The loss function used for PlaneRCNN is the same as what is used in vanilla PlaneRCNN. 
 
#### Putting it all together
The final loss was computed as a weighted sum of all the average of three individual losses.         
formula: ![img](<formula.PNG>)

While training it has been observed that PlaneRCNN cannot be trained with a batch size of more than 1. So the entire training process was conducted using batch size of 1. This took a lot of training time on Colab GPU. For training on more than batch size one, from my understanding it can be trained as a part training (train YOLO then PlaneRCNN). 

train: ![img](<train.PNG>)

### Validation - Output
YOLO output: ![img](</images_readme/YoloOut.png>)
MiDaS output: ![img](</images_readme/midas.png>)
