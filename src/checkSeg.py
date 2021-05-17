from PIL import Image
import os
import sys
import numpy as np
from natsort import natsorted
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
import glob
import shutil

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
import albumentations as albu

import random
random.seed(10)
#np.random.seed(seed=25)
#np.random.seed(seed=40)
np.random.seed(seed=30)

from semanticSeg import Dataset
from semanticSeg import get_training_augmentation
from semanticSeg import get_preprocessing
from semanticSeg import get_validation_augmentation
from semanticSeg import visualize


#x_valid_dir = './image/train/valid_images/'
#y_valid_dir = './image/train/valid_masks/'
x_valid_dir = './image/train/valid_images2/'
y_valid_dir = './image/train/valid_masks2/'
CLASSES = ['background', 'building-flooded', 'building-non-flooded', 'road-flooded', 'road-non-flooded', 'water', 'tree', 'vehicle', 'pool', 'grass']
DEVICE = 'cuda'
#ENCODER = 'resnet50'
ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]


# load best saved checkpoint
#best_model = torch.load('./best_model_Unet_resnet50_final.pth')
best_model = torch.load('./best_model_Unet_resnet50_final_plusplus.pth')
#print(best_model)


# create test dataset
test_dataset = Dataset(
    x_valid_dir, #x_test_dir
    y_valid_dir, #y_test_dir
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)

# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

#logs = test_epoch.run(test_dataloader)


# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_valid_dir, y_valid_dir, #x_test_dir, y_test_dir,
    classes=CLASSES,
)

for i in range(20):
    n = np.random.choice(len(test_dataset))

    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]

    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    gt_mask = np.transpose(gt_mask, (1, 2, 0))
    pr_mask = np.transpose(pr_mask, (1, 2, 0))

    gt_mask_gray = np.zeros((gt_mask.shape[0],gt_mask.shape[1]))

    for j in range(gt_mask.shape[2]):
        gt_mask_gray = gt_mask_gray + 1/gt_mask.shape[2]*j*gt_mask[:,:,j]

    pr_mask_gray = np.zeros((pr_mask.shape[0],pr_mask.shape[1]))
    for j in range(pr_mask.shape[2]):
        pr_mask_gray = pr_mask_gray + 1/pr_mask.shape[2]*j*pr_mask[:,:,j]

    gt_mask_all = gt_mask[..., 0].squeeze()*0
    pr_mask_all = pr_mask[..., 0].squeeze()*0
    for j in range(1,10):
        gt_mask_all = gt_mask_all + gt_mask[..., j].squeeze()*j
        pr_mask_all = pr_mask_all + pr_mask[..., j].squeeze()*j

    visualize(
        image=image_vis, 
        #ground_truth_mask=gt_mask_gray, 
        #predicted_mask=pr_mask_gray
        ground_truth_mask=gt_mask_all,
        predicted_mask = pr_mask_all
    )