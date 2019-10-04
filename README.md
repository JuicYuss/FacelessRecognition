# Faceless Recognition
Challenge from DataHack2019 on person recognition in low resolution images: https://www.datahack.org.il/challenge/orcam-faceless-recognition

**This is not the last version of the project.

### Data & results

The data set includes 500k+ samples.
A sample is presented as follows: 
FacelessRecognition/data/person_0000/video_0000/seq_0000/frame_0000

Each frame is of resolution 64 * 64. 
The label of a frame is the "person####" folder name.

Our training/test method is to devide the data randomly and evenly between the persons we want to classify => implementing SRGAN for adding new features => training our CNN model on it. 

We managed to achieve accuracy of ~80% and more, and Loss ~0.15 and less on the test data, which is composed of 24 frames per 2 random sequences per 2 random videos of a person. 

## Training Process
### 1. SRGAN
We used SRGAN (super resolution GAN) to increase the resolution of the images. It enhances the original image of 64 * 64 pixels
into 224 * 224 pixel image.

![img1](https://github.com/yussiroz/FacelessRecognition/blob/master/samples/example_1.jpg)![img2](https://github.com/yussiroz/FacelessRecognition/blob/master/samples/example_2.jpg)

Reference to the source : https://github.com/krasserm/super-resolution

### 2. CNN
We trained a CNN on the super resolution images.

<img src="https://github.com/yussiroz/FacelessRecognition/blob/master/samples/model.png" width = 400 hight = 800>

model weights: https://drive.google.com/file/d/1npsTYjmgLw78aTGnxlU8h5wHuxtL2du4/view?usp=sharing

## Prerequisites 
python = 3.5.6, tensorflow = 1.10.0, tensorflow-gpu = 2.0.0rc0
