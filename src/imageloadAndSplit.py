from PIL import Image
import os
import sys
import numpy as np
from natsort import natsorted
import cv2
import matplotlib.pyplot as plt
import glob


# splited image size
height = 1000
width = 1000
height_origin = 3000
width_origin = 4000
left_c = 0
top_c = 0
right_c = 4000
bottom_c = 3000


#load data
x_train_files_f_tmp = glob.glob('./image/train/flooded/images/*')
y_train_files_f_tmp = glob.glob('./image/train/flooded/mask/*')

x_train_files_n_tmp = glob.glob('./image/train/non-flooded/images/*')
y_train_files_n_tmp = glob.glob('./image/train/non-flooded/mask/*')

x_train_files_f = natsorted(x_train_files_f_tmp)
y_train_files_f = natsorted(y_train_files_f_tmp)
x_train_files_n = natsorted(x_train_files_n_tmp)
y_train_files_n = natsorted(y_train_files_n_tmp)


idx = 1

im_image = Image.open(x_train_files_f[idx])
im_list_image = np.asarray(im_image)

im_mask = Image.open(y_train_files_f[idx])
im_list_mask = np.asarray(im_mask)

#print(im_list_mask)
#print(len(im_list_mask))
print(im_list_image.shape)
print(im_list_image.shape[0])
print(im_list_image.shape[1])

'''
im_width, im_height = im_image.size
if im_height > 3000:
    im = im_image.crop((left_c, top_c, right_c, bottom_c))
else:
    im = im_image

im_list = np.asarray(im)
print(im_list.shape)

plt.imshow(im_list_image)
plt.show()


plt.imshow(im_list)
plt.show()
'''

#plt.imshow(im_list_mask)
#plt.show()



def ImgSplit(im):
    buff = []
    for h1 in range(int(height_origin/height)):
        for w1 in range(int(width_origin/width)):
            w2 = w1 * height
            h2 = h1 * width
            #print(w2, h2, width + w2, height + h2)
            c = im.crop((w2, h2, width + w2, height + h2))
            buff.append(c)
    return buff


'''
x_train_files = x_train_files_f
y_train_files = y_train_files_f
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
        ig.save("./image/train/train_images_f/"+ str(file_name) + '_' + str(i) +'_' + str(hi) +".jpg")


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
        ig.save("./image/train/train_masks_f/"+ str(file_name) + '_' + str(i) +'_' + str(hi) +".png")

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