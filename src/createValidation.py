from PIL import Image
import os
import sys
import numpy as np
from natsort import natsorted
import cv2
import matplotlib.pyplot as plt
import glob
import shutil

#load train data
x_train_files_f = glob.glob('./image/train/train_images_f/*')
y_train_files_f = glob.glob('./image/train/train_masks_f/*')
x_train_files_n = glob.glob('./image/train/train_images_n/*')
y_train_files_n = glob.glob('./image/train/train_masks_n/*')

#load validation data
x_valid_files_f = glob.glob('./image/train/valid_images_f/*')
y_valid_files_f = glob.glob('./image/train/valid_masks_f/*')
x_valid_files_n = glob.glob('./image/train/valid_images_n/*')
y_valid_files_n = glob.glob('./image/train/valid_masks_n/*')


# sort
X_train_files_f = natsorted(x_train_files_f)
Y_train_files_f = natsorted(y_train_files_f)
X_train_files_n = natsorted(x_train_files_n)
Y_train_files_n = natsorted(y_train_files_n)

X_valid_files_f = natsorted(x_valid_files_f)
Y_valid_files_f = natsorted(y_valid_files_f)
X_valid_files_n = natsorted(x_valid_files_n)
Y_valid_files_n = natsorted(y_valid_files_n)

idx = 41

im_image = Image.open(X_train_files_f[idx])
im_list_image = np.asarray(im_image)

im_mask = Image.open(Y_train_files_f[idx])
im_list_mask = np.asarray(im_mask)

plt.imshow(im_list_image, vmin=0, vmax=9)
plt.show()

plt.imshow(im_list_mask, vmin=0, vmax=9)
plt.show()