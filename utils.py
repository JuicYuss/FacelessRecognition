import numpy as np
from common import resolve_single
from PIL import Image
import os
from tqdm import tqdm
from scipy.stats import mode
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import cv2

def load_image(path):
    return np.array(Image.open(path))

def per_label(y, top_five):
   l = []
   pred=[]
   for col in range(y.shape[1]):
       a = np.argwhere(y[:,col] == 1).reshape(5)
       l.append((col, a))
       top = mode(top_five[a], 0)
       pred.append((col,top[0]))
   return pred,l


def train_model(iterations, name, model, srgan, resize = False):
    filepath="weights/custom/" + name + ".h5"
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    for i in range(iterations):
        train_set = devide(24, 2, 2)
        X = tensor2numpy('./data/', train_set, srgan)
        if(resize):
            for k, v in X.items():
                X[k] = cv2.resize(v, (224,224))
        x = [X[i] for i in X.keys()]
        train = np.array(x, dtype = "float64")
        y = create_onehot(X)
        history = model.fit(train, y, batch_size=32, epochs=7, callbacks=callbacks_list, validation_split=0.2)
        # Plot training & validation accuracy values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


def devide_submission(frs):
    directories = [d for d in os.listdir('./data/') if d[:2]=='se']
    frames = []
    for d in directories:
        fram = shuffle([d for d in os.listdir('./data/' + d )], frs)
        for ff in fram:
            frames.append('{}/{}'.format(d, ff))
    return frames

#def plot_sample(lr, sr):
#    plt.figure(figsize=(20, 10))
#
#    images = [lr, sr]
#    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']
#
#    for i, (img, title) in enumerate(zip(images, titles)):
#        plt.subplot(1, 2, i+1)
#        plt.imshow(img)
#        plt.title(title)
#        plt.xticks([])
#        plt.yticks([])

def tensor2numpy(root, files, model):
    sr_img = {}
    for f in tqdm(files):
        directory = os.path.join(root, f)
        img = load_image(directory)
        n = resolve_single(model, img)
        sr_img[f] = n.numpy()
    return sr_img

def shuffle(x, i):
    np.random.shuffle(x)
    return x[:i]

def devide(frs, sequences, videos):
    directories = [d for d in os.listdir('./data/') if d[:2]=='pe']
    frames = []
    for d in directories:
        vids = shuffle([d for d in os.listdir('./data/' + d + '/')], videos)
        for v in vids:
            seqs = shuffle([d for d in os.listdir('./data/' + d + '/' + v + '/')], sequences)
            for f in seqs:
                fr = shuffle([d for d in os.listdir('./data/' + d + '/' + v + '/' + f + '/')], frs)
                for ff in fr:
                    frames.append('{}/{}/{}/{}'.format(d,v,f,ff))
    return frames



def create_onehot(data):
    n = len(data)
    cols = 101
    y = np.zeros((n,cols))
    counter = 0
    for i in data.keys():
        my_slash=i.split('/')
        my_underscore=[i.split('_')[1] for i in my_slash]
        person = int(my_underscore[0])
        y[counter,person] = 1
        counter += 1
    return y

def enumerate_paths(paths):
    # Extract sequences/videos/people from the frame-paths
    sequences = [os.path.dirname(p) for p in paths]
    videos = [os.path.dirname(s) for s in sequences]
    people = [os.path.dirname(c) for c in videos]

    # Enumerate the frames based on videos and people
    unique_videos, video_ids = np.unique(videos, return_inverse=True)
    unique_people, person_ids = np.unique(people, return_inverse=True)
    return person_ids, video_ids


def split_by(data, indices):
    # Split data based on a numpy array of sorted indices
    sections = np.where(np.diff(indices))[0] + 1
    split_data = np.split(data, sections)
    return split_data

