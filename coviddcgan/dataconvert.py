from PIL import Image
import os, sys
import cv2
import numpy as np
import random

'''
Converts image dataset to format usable in machine learning
Also resizes the images for faster performance
'''

# Path to image directory
path_covid_train = "train/covid/"
path_normal_train = "train/normal/"
path_covid_test = "test/covid/"
path_normal_test = "test/normal/"

covid_train_dirs = os.listdir( path_covid_train )
covid_train_dirs.sort()
normal_train_dirs = os.listdir( path_normal_train )
normal_train_dirs.sort()
covid_test_dirs = os.listdir( path_covid_test )
covid_test_dirs.sort()
normal_test_dirs = os.listdir( path_normal_test )
normal_test_dirs.sort()

x_train_covid = []
y_train_covid = []
x_test_covid = []
y_test_covid = []

x_train_normal = []
y_train_normal = []
x_test_normal = []
y_test_normal = []

def load_dataset():
    # Append images to a list
    for item in covid_train_dirs:
        if os.path.isfile(path_covid_train+item):
            im = Image.open(path_covid_train+item).convert("L")
            im = im.resize((128,128))
            im = np.asarray(im)
            im = im.astype(np.uint8)
            x_train_covid.append(im)
            y_train_covid.append(0)
    for item in normal_train_dirs:
        if os.path.isfile(path_normal_train+item):
            im = Image.open(path_normal_train+item).convert("L")
            im = im.resize((128,128))
            im = np.asarray(im)
            im = im.astype(np.uint8)
            x_train_normal.append(im)
            y_train_normal.append(1)
    for item in covid_test_dirs:
        if os.path.isfile(path_covid_test+item):
            im = Image.open(path_covid_test+item).convert("L")
            im = im.resize((128,128))
            im = np.asarray(im)
            im = im.astype(np.uint8)
            x_test_covid.append(im)
            y_test_covid.append(0)
    for item in normal_test_dirs:
        if os.path.isfile(path_normal_test+item):
            im = Image.open(path_normal_test+item).convert("L")
            im = im.resize((128,128))
            im = np.asarray(im)
            im = im.astype(np.uint8)
            x_test_normal.append(im)
            y_test_normal.append(1)

if __name__ == "__main__":

    load_dataset()
    
    # Save the images in '.npz' directories
    np.savez("dataset_covid.npz", x_train=x_train_covid, y_train=y_train_covid, x_test=x_test_covid, y_test=y_test_covid)
    np.savez("dataset_normal.npz", x_train=x_train_normal, y_train=y_train_normal, x_test=x_test_normal, y_test=y_test_normal)
