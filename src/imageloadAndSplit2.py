from PIL import Image
import os
import sys
import numpy as np
from natsort import natsorted
import cv2
import matplotlib.pyplot as plt
import glob


#resize
left_c = 0
top_c = 0
right_c = 992
bottom_c = 992


#load data
x_train_files_f_tmp = glob.glob('./image/train/resized/train_images_f/*')
y_train_files_f_tmp = glob.glob('./image/train/resized/train_masks_f/*')
x_train_files_n_tmp = glob.glob('./image/train/resized/train_images_n/*')
y_train_files_n_tmp = glob.glob('./image/train/resized/train_masks_n/*')


x_valid_files_f_tmp = glob.glob('./image/train/resized/valid_images_f/*')
y_valid_files_f_tmp = glob.glob('./image/train/resized/valid_masks_f/*')
x_valid_files_n_tmp = glob.glob('./image/train/resized/valid_images_n/*')
y_valid_files_n_tmp = glob.glob('./image/train/resized/valid_masks_n/*')


x_train_files_f = natsorted(x_train_files_f_tmp)
y_train_files_f = natsorted(y_train_files_f_tmp)
x_train_files_n = natsorted(x_train_files_n_tmp)
y_train_files_n = natsorted(y_train_files_n_tmp)

x_valid_files_f = natsorted(x_valid_files_f_tmp)
y_valid_files_f = natsorted(y_valid_files_f_tmp)
x_valid_files_n = natsorted(x_valid_files_n_tmp)
y_valid_files_n = natsorted(y_valid_files_n_tmp)


x_train_files = x_train_files_n
y_train_files = y_train_files_n
for i in range(len(x_train_files)):
    im_tmp=Image.open(x_train_files[i])
    file_name = os.path.splitext(os.path.basename(x_train_files[i]))[0]

    hi=0
    im = im_tmp.crop((left_c, top_c, right_c, bottom_c))
    im.save("./image/train/resized/train_images2/"+ str(file_name) + '_resized' +".jpg")


for i in range(len(y_train_files)):
    im_tmp=Image.open(y_train_files[i])
    file_name = os.path.splitext(os.path.basename(y_train_files[i]))[0]

    hi=0
    im = im_tmp.crop((left_c, top_c, right_c, bottom_c))
    im.save("./image/train/resized/train_masks2/"+ str(file_name) + '_resized' +".png")


x_train_files = x_valid_files_n
y_train_files = y_valid_files_n
for i in range(len(x_train_files)):
    im_tmp=Image.open(x_train_files[i])
    file_name = os.path.splitext(os.path.basename(x_train_files[i]))[0]

    hi=0
    im = im_tmp.crop((left_c, top_c, right_c, bottom_c))
    im.save("./image/train/resized/valid_images2/"+ str(file_name) + '_resized' +".jpg")


for i in range(len(y_train_files)):
    im_tmp=Image.open(y_train_files[i])
    file_name = os.path.splitext(os.path.basename(y_train_files[i]))[0]

    hi=0
    im = im_tmp.crop((left_c, top_c, right_c, bottom_c))
    im.save("./image/train/resized/valid_masks2/"+ str(file_name) + '_resized' +".png")



'''
x_train_files = x_train_files_n
y_train_files = y_train_files_n
for i in range(len(x_train_files)):
    im_tmp=Image.open(x_train_files[i])
    file_name = os.path.splitext(os.path.basename(x_train_files[i]))[0]

    hi=0
    im_width, im_height = im_tmp.size
    if im_height > 3000:
        im = im_tmp.crop((left_c, top_c, right_c, bottom_c))
    else:
        im = im_tmp
    for ig in ImgSplit(im):
        hi=hi+1
        ig.save("./image/train/train_images_n/"+ str(file_name) + '_' + str(i) +'_' + str(hi) +".jpg")


for i in range(len(y_train_files)):
    im_tmp=Image.open(y_train_files[i])
    file_name = os.path.splitext(os.path.basename(y_train_files[i]))[0]

    hi=0
    im_width, im_height = im_tmp.size
    if im_height > 3000:
        im = im_tmp.crop((left_c, top_c, right_c, bottom_c))
    else:
        im = im_tmp
    for ig in ImgSplit(im):
        hi=hi+1
        ig.save("./image/train/train_masks_n/"+ str(file_name) + '_' + str(i) +'_' + str(hi) +".png")
'''