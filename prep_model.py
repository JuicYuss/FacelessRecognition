#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:19:33 2019

@author: yussiroz
"""
from tensorflow.keras.callbacks import ModelCheckpoint
import data
from data import Images
from common import resolve_single
import matplotlib.pyplot as plt
from scipy.stats import mode
"""Upload super res GAN libs"""
from model.srgan import generator
from model.cnn import CNN
from utils import tensor2numpy, shuffle, devide, create_onehot, per_label, devide_submission
import numpy as np
from sklearn.preprocessing import OneHotEncoder

"""Upload images, poses, signatures"""
poses = data.read_pose('./data/pose.pkl')
signatures = data.read_signatures('./data/signatures.pkl')
with Images('data/images.tar') as images:
    path = images.paths[20000]
    image = images._getitem(path)
    print ('read image {} of shape {}'.format(path, image.shape))

"""Use SRGAN"""
model = generator()
model.load_weights('weights/srgan/gan_generator.h5')

my_split=poses[0]
my_split=[path[:-4] for path in my_split]

"""Upload customed cnn model"""
cnn = CNN(256, 256, 3, 101)
cnn.load_weights('./cnn_weights.h5')

filepath="./cnn_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

"""Prepare and train on a batch of data and labels, 10 iterations"""
for i in range(10):
    train_set = devide(24, 2, 2)
    X = tensor2numpy('./data/', train_set, model)
    x = [X[i] for i in X.keys()]
    train = np.array(x, dtype = "float64")
    y = create_onehot(X)
    cnn.fit(train, y, batch_size=32, epochs=5, callbacks=callbacks_list)

"""Make predictions"""
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




#submitter = "My Awesome Team Name"
#
#from urllib import request
#import json
#
#jsonStr = json.dumps({'submitter': submitter, 'predictions': predictions})
#data = jsonStr.encode('utf-8')
#req = request.Request('https://leaderboard.datahack.org.il/orcam/api',
#                  headers={'Content-Type': 'application/json'},
#                  data=data)
#resp = request.urlopen(req)
