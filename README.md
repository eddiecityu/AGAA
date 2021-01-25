# AGAA (Auto-Gen Architectural Analytics)

# Detection of Objects in a Floor Plan and Architectural Images

There are few models available for doing object detection recognition in an image. Like RCNN, fast RCNN, faster RCNN, mask RCNN, Yolo, SSD etc. all of them are developed and configured for natural images. In this project we are working on document images of floor plans. In a floor plan image, we have objects like dining table, sofa, sink, etc.

we used the yolo and faster RCNN for object detection.

used darkflow implementation of yolo https://github.com/thtrieu/darkflow
used https://github.com/kbardool/keras-frcnn for frcnn

# REQUIREMENTS-
- Numpy, pandas, matplotlib
- opencv version 3.0 or above
- tensorflow 1.10
- Cython 0.28.2
- Tkinter
- CUDA

# DATASET PREPARATION (GENERATING IMAGE ANNOTATION)-
1. Image annotaion is the most time consuming task. annotate each object in an image and store their coordinates and label in an xml file.
2. You have to annotate each image manually. you can use some tools for annotation and you would be able to annotate at best 30 images in an hour. 

Data Labelling Tools: https://github.com/wkentaro/labelme

Labelme is a graphical image annotation tool inspired by http://labelme.csail.mit.edu.
It is written in Python and uses Qt for its graphical interface.

Windows
Install Anaconda, then in an Anaconda Prompt run:

1. python3
2. conda create --name=labelme python=3.6
3. conda activate labelme
4. pip install labelme

# YOLO-

1. SETTING UP

download the darkflow yolo from above given link.
download weight, cfg files from https://pjreddie.com/darknet/yolo/ there are plenty available and download the one which you need and add them to downloaded darkflow repository in bin and cfg directories respectively.
alternatively you can also download our pretrained model and weight if you want use them from this link https://drive.google.com/drive/u/1/folders/1rtOYXL1f8m3Ffwbj-_0aIiWBFMW9qjjT
download both folder ckpt and cfg and add them in the main directory of darkflow.
download dataset from above links or you can use your own and addd them to dataset repository in downloaded darkflow repositor.
TRAINING

assume that you want to use tiny-yolo cfg for training

create a copy of configuration file tiny-yolo-voc.cfg and rename it to tiny-yolo-voc-12c.cfg (12c refer to the number of objects or classes we are identifying ) leave the original file unchanged.

In tiny-yolo-voc-12c.cfg change classes in the [region] layer (the last layer) to the number of classes you are going to train for. in our case, it is 12.

change filters in the [convolutional] layer (the second to last layer) to num * (classes + 5). In our case, num is 5 and classes are 12 so 5 * (12 + 5) = 85 therefore filters are set to 85.

Change labels.txt to include the label(s) you want to train on. In our case, labels.txt will contain 12 labels.

To train your the model you can run the command-
python flow --model cfg/tiny-yolo-voc-3c.cfg --load bin/tiny-yolo-voc.weights --train --annotation dataset/train_annotation --dataset dataset/train_images

2. PREDICTING

to predict a single image set the image file path in predict_img.py and run it.
you can prefer to change in option field like model, load values, epochs, etc.

to evalute the images of test set use command-
python flow --imgdir dataset/test_images --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights --json
output would be store in json format in dataset/test_images/out directory

# FASTER RCNN-

1. SETTING UP

download the frcnn folder from above and set file image directory path.
convert xml annotation of train images into text file, for this you can have look at
https://www.analyticsvidhya.com/blog/2018/11/implementation-faster-r-cnn-python-object-detection/
2. TRAINING

to train model run command-
python train_frcnn.py -o simple -p train.txt

3. TESTING

to test model run command-
python test_frcnn.py -p test_images

4. RESULTS

YOLO- object detected and accuracy achieved
