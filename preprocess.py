#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:19:33 2019

@author: yussiroz
"""
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Dropout, GlobalAveragePooling2D, Conv2D, concatenate, GlobalMaxPooling2D,
import data
from data import Images
import matplotlib.pyplot as plt
from scipy.stats import mode
"""Upload super res GAN libs"""
from model.srgan import generator
from utils import tensor2numpy, shuffle, devide, create_onehot, per_label, devide_submission
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.nasnet import NASNetMobile
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

"""Upload images, poses, signatures"""
poses = data.read_pose('./data/pose.pkl')
signatures = data.read_signatures('./data/signatures.pkl')
with Images('data/images.tar') as images:
    path = images.paths[0]
    image = images._getitem(path)
    print ('read image {} of shape {}'.format(path, image.shape))


"""Use SRGAN"""
model = generator()
model.load_weights('weights/srgan/gan_generator.h5')
paths = poses[0]

my_split=poses[0]
my_split=[path[:-4] for path in my_split]


def CNN(shape1, shape2, shape3, nbClasses):
    input_shape = (shape1, shape2, shape3)
#    optimizer = Adam(0.005, beta_1=0.1, beta_2=0.001, amsgrad=True)
    n_classes = nbClasses
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()
    return model


cnn = CNN(256, 256, 3, 101)
cnn.load_weights('./cnn_weights.h5')

filepath="./cnn_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


for i in range(10):
    train_set = devide(24, 2, 2)
    X = tensor2numpy('./data/', train_set, model)
    x = [X[i] for i in X.keys()]
    train = np.array(x, dtype = "float64")
    y = create_onehot(X)
    cnn.fit(train, y, batch_size=32, epochs=5, callbacks=callbacks_list)


sss = devide_submission(5)
preds = {}
for i in range(0, len(sss),5):
    test_set = tensor2numpy('./data/', sss[i:i+5], model)
    t = [test_set[i] for i in test_set.keys()]
    test = np.array(t, dtype = 'float64')
    prediction = cnn.predict(test)
    top_five = np.argsort(prediction, axis=1)[:,:5]
    top = mode(top_five,0)
    preds[sss[i]] = top


top_five_prediction = {}
for k, v in preds.items():
    top_five_prediction[k] = v[0]


l = [0] * len(preds)
for k in preds.keys():
    my_seq=int(k[4:8])
    l[my_seq]=top_five_prediction[k].tolist()

predictions = [i[0] for i in l]




submitter = "My Awesome Team Name"

from urllib import request
import json

jsonStr = json.dumps({'submitter': submitter, 'predictions': predictions})
data = jsonStr.encode('utf-8')
req = request.Request('https://leaderboard.datahack.org.il/orcam/api',
                  headers={'Content-Type': 'application/json'},
                  data=data)
resp = request.urlopen(req)
