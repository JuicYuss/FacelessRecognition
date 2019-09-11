import numpy as np
import matplotlib.pyplot as plt
from common import resolve_single
from PIL import Image
import os
from tqdm import tqdm

def load_image(path):
    return np.array(Image.open(path))


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


