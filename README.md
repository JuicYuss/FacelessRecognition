# Faceless Recognition
Challenge from DataHack2019 on person recognition in low resolution images.

The data set included 500k+ samples.

## Training Process
### 1. SRGAN
We used SRGAN (super resolution GAN) to increase the resolution of the images. 

![img1](https://github.com/yussiroz/FacelessRecognition/blob/master/samples/example_1.jpg)![img2](https://github.com/yussiroz/FacelessRecognition/blob/master/samples/example_2.jpg)

### 2. CNN
We trained a CNN on the super resolution images.



## Prerequisites 
python = 3.5.6, tensorflow = 1.10.0, tensorflow-gpu = 2.0.0rc0
