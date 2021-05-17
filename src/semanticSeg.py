from PIL import Image
import os
import sys
import numpy as np
from natsort import natsorted
import cv2
import matplotlib.pyplot as plt
import glob
import shutil

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
import albumentations as albu

import random
random.seed(10)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if not torch.cuda.is_available():
    raise Exception("GPU is not available.")
else:
    pass
#print("device name", torch.cuda.get_device_name(0))


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)

    """

    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    CLASSES = ['background', 'building-flooded', 'building-non-flooded', 'road-flooded', 'road-non-flooded', 'water', 'tree', 'vehicle', 'pool', 'grass']

    def __init__(
            self, 
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids_tmp = os.listdir(images_dir)
        self.ids_mask_tmp = os.listdir(masks_dir)
        self.ids = natsorted(self.ids_tmp)
        self.ids_mask = natsorted(self.ids_mask_tmp)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.ids_mask]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)


        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# visualizer
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, vmin=0, vmax=9)
    plt.show()


'''
x_train_dir = './image/train/train_images_f/'
y_train_dir = './image/train/train_masks_f/'

dataset = Dataset(x_train_dir, y_train_dir, classes=['background', 'building-flooded', 'building-non-flooded', 'road-flooded', 'road-non-flooded', 'water', 'tree', 'vehicle', 'pool', 'grass'])
#dataset = Dataset(x_train_dir, y_train_dir, classes=['building-non-flooded'])
#dataset = Dataset(x_train_dir, y_train_dir, classes=['grass'])
#dataset = Dataset(x_train_dir, y_train_dir, classes=['building-non-flooded', 'grass'])

image, mask = dataset[41] # get some sample

visualize(
    image=image, 
    background=mask[..., 0].squeeze(),
    building_f=mask[..., 1].squeeze(),
    building=mask[..., 2].squeeze(),
    road_f=mask[..., 3].squeeze(),
    road=mask[..., 4].squeeze(),
    water=mask[..., 5].squeeze(),
    tree=mask[..., 6].squeeze(),
    vehicle=mask[..., 7].squeeze(),
    pool=mask[..., 8].squeeze(),
    grass=mask[..., 9].squeeze(),
    #background_mask=mask[..., 10].squeeze(),
)
'''


'''
x_train_dir = './image/train/train_images_f/'
y_train_dir = './image/train/train_masks_f/'

augmented_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    classes=['background', 'building-flooded', 'building-non-flooded', 'road-flooded', 'road-non-flooded', 'water', 'tree', 'vehicle', 'pool', 'grass'],
)

# same image with different random transforms
for i in range(3):
    image, mask = augmented_dataset[41]
    #visualize(image=image, mask=mask.squeeze(-1))
    visualize(
    image=image, 
    background=mask[..., 0].squeeze(),
    building_f=mask[..., 1].squeeze(),
    building=mask[..., 2].squeeze(),
    road_f=mask[..., 3].squeeze(),
    road=mask[..., 4].squeeze(),
    water=mask[..., 5].squeeze(),
    tree=mask[..., 6].squeeze(),
    vehicle=mask[..., 7].squeeze(),
    pool=mask[..., 8].squeeze(),
    grass=mask[..., 9].squeeze(),
    #background_mask=mask[..., 10].squeeze(),
)
'''



if __name__ == '__main__':
    #create model
    ENCODER = 'resnet50'
    #ENCODER = 'efficientnet-b2'
    #ENCODER = 'efficientnet-b4'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['background', 'building-flooded', 'building-non-flooded', 'road-flooded', 'road-non-flooded', 'water', 'tree', 'vehicle', 'pool', 'grass']
    ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'

    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS, 
        classes=n_classes, 
        activation=ACTIVATION,
    )
    
    ## create segmentation model with pretrained encoder
    #model = smp.UnetPlusPlus(
    #    encoder_name=ENCODER, 
    #    encoder_weights=ENCODER_WEIGHTS, 
    #    classes=n_classes, #len(CLASSES), 
    #    activation=ACTIVATION,
    #)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)



    # load data
    #x_train_dir = './image/train/train_images_f/'
    #y_train_dir = './image/train/train_masks_f/'
    #x_valid_dir = './image/train/valid_images_f/'
    #y_valid_dir = './image/train/valid_masks_f/'

    x_train_dir = './image/train/train_images/'
    y_train_dir = './image/train/train_masks/'
    x_valid_dir = './image/train/valid_images2/'
    y_valid_dir = './image/train/valid_masks2/'


    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=get_validation_augmentation(), 
        #augmentation=None,
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    #valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])


    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )



    ## train model
    max_score = 0

    # for visualization
    x_epoch_data = []
    train_dice_loss = []
    train_iou_score = []
    valid_dice_loss = []
    valid_iou_score = []


    for i in range(0, 10):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        x_epoch_data.append(i)
        train_dice_loss.append(train_logs['dice_loss'])
        train_iou_score.append(train_logs['iou_score'])
        valid_dice_loss.append(valid_logs['dice_loss'])
        valid_iou_score.append(valid_logs['iou_score'])

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model_Unet_resnet50_final_2.pth')
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    line1, = ax1.plot(x_epoch_data,train_dice_loss,label='train') 
    line2, = ax1.plot(x_epoch_data,valid_dice_loss,label='validation')
    ax1.set_title("dice loss")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('dice_loss')
    ax1.legend(loc='upper right')

    ax2 = fig.add_subplot(1, 2, 2)
    line1, = ax2.plot(x_epoch_data,train_iou_score,label='train')
    line2, = ax2.plot(x_epoch_data,valid_iou_score,label='validation') 
    ax2.set_title("iou score")
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('iou_score')
    ax2.legend(loc='upper left')

    plt.show()
