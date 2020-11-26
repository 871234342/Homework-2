# HOMEWORK 2 - Street View House Number Object Detection

This model is for object detection for the Street Veiw House Numbers dataset.
This repo is based on this [repo](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch).

### Hardware
- Ubuntu 18.04.5 LTS
- Intel® Xeon® Silver 4210 CPU @ 2.20GHz
- NVIDIA GeForce RTX 2080 Ti

### Reproduce Submission
To reproduce my submission without training, do the following:
1. [Installation](#Installation)
2. [Data Preparation](#Data-Preparation)
3. [Inference](#Inference)


### Installation
Install all the requirments.

`pip install pycocotools numpy opencv-python tqdm pyymal webcolors torch torchvision`


### Data Preparation
The data should be placed as follows:
```
repo
  +- train
  |  +- 1.png
  |  +- 2.png
  |  +- ...
  |
  +- val
  |  +- 32403.png
  |  +- 32404.png
  |  +- ...
  |
  +- test
  |  +- 1.png
  |  +- 2.png
  |  +- ...
  |
  +- annotation_train.csv
  +- annotation_val.csv
  +- train.py
  +- infer.py
  +- weights.pth   (needed for inference)
  +- numbers.yml   (needed for training)
  |  ...
```
Where train folder contains all the training images, val folder contains all the validation images, and test folder contains all test images. Both annotation_train.csv and annotaion_val.csv should contain the file name, corresponding labels and bounding boxes of each image in train and val folder. Please check annotaion_train.csv to see the expected format.

### Training
To train, please download the pretrained weight [here](https://drive.google.com/file/d/16JwAXozDpqjoWRE1MySUKH0JdkstXMqf/view?usp=sharing) and put it beside train.py. Simply run train.py. The weights should be saved in 'logs/numbers' folder. There will be several file saving weights at different training process. The batch_size is set to be 3. Make it smaller in numbers.yml if memory is not sufficent.

### Inference
for inference, please download the weights file [here](https://drive.google.com/file/d/1t9W1wxUjfBOTtGo0I7-tlpgdlKqvSM01/view?usp=sharing) and put it beside infer.py. Simply run infer.py and predictions.json containing images file names and their corresponding predictions will be created.

### Citation
[Yet Anothor EfficientDet Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
