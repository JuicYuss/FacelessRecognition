"""
Created on Tue Sep  3 14:19:33 2019

@author: YussiRoz, InbalWeiss

"""
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import NASNetMobile, VGG19
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D
from tensorflow.keras.models import Model
import data
from data import Images
from common import resolve_single
import matplotlib.pyplot as plt
from scipy.stats import mode
"""Upload super res GAN libs"""
from model.srgan import generator
from model.cnn import CNN
from utils import tensor2numpy, shuffle, devide, create_onehot, per_label, devide_submission, train_model
import numpy as np
from sklearn.preprocessing import OneHotEncoder

"""Upload images, poses, signatures"""
poses = data.read_pose('./data/pose.pkl')
signatures = data.read_signatures('./data/signatures.pkl')
with Images('data/images.tar') as images:
    path = images.paths[20000]
    image = images._getitem(path)
    print ('read image {} of shape {}'.format(path, image.shape))


my_split=poses[0]
my_split=[path[:-4] for path in my_split]


"""Use SRGAN"""
srgan = generator()
srgan.load_weights('weights/srgan/gan_generator.h5')


"""Upload customed cnn model"""
cnn = CNN(256, 256, 3, 101)
cnn.load_weights('weights/custom/cnn_plus.h5')
plot_model(cnn, to_file='./model.png', show_shapes=True, show_layer_names=True)


train_model(2, 'cnn_plus', cnn, srgan)

#filepath="./cnn_weights.h5"
#checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]

"""Prepare and train on a batch of data and labels, 10 iterations"""
for i in range(2):
    train_set = devide(24, 2, 2)
    X = tensor2numpy('./data/', train_set, srgan)
    x = [X[i] for i in X.keys()]
    train = np.array(x, dtype = "float64")
    y = create_onehot(X)
    history = cnn.fit(train, y, batch_size=32, epochs=5, callbacks=callbacks_list, validation_split=0.2)
    # Plot training & validation accuracy values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

"""Upload, use transfer learning"""
VGG=VGG19(input_shape=(224,224,3),include_top=False,weights='imagenet')
#freeze NatNetMobile weights
for layer in VGG.layers:
    layer.trainable=False

output=VGG.layers[6].output
output=Flatten()(output)
output=Dense(1024,activation='relu')(output)
output=Dense(512,activation='relu')(output)
output=Dropout(0.5)(output)
output=Dense(256,activation='relu')(output)
output=Dense(101,activation='softmax')(output)
VGG=Model(VGG.input,output)
VGG.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
plot_model(VGG, to_file='./model.png', show_shapes=True, show_layer_names=True)

train_model(3, 'vgg', VGG, srgan, resize = True)

"""Test, evaluate the model"""
test_set = devide(12, 2, 2)
X = tensor2numpy('./data/', test_set, srgan)
x = [X[i] for i in X.keys()]
test = np.array(x, dtype = "float64")
y = create_onehot(X)
evaluation = cnn.evaluate(test, y)
prediction = cnn.predict(test)

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
