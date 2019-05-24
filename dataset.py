import os
import numpy as np
import cv2
from config import cfg
import random
import time

def Preprocessing(pose_img, d, stage='test'):

    height, width = cfg.data_shape

    imgs = []

    vis = False

    img = pose_img

    while img is None:
        print('read none image')

    # max = 1280
    add = max(img.shape[0], img.shape[1])

    # image border, bimg : 3280(720) x 3840(1280)
    bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT,
                              value=cfg.pixel_means.reshape(-1))

    #bbox : x, y, w, h
    #bbox = np.array(d['bbox']).reshape(4, ).astype(np.float32)
    bbox = np.array(d[:4]).reshape(4, ).astype(np.float32)

    # x + 1280, y + 1280
    bbox[:2] += add

    crop_width = bbox[2] * (1 + cfg.imgExtXBorder * 2)
    crop_height = bbox[3] * (1 + cfg.imgExtYBorder * 2)

    objcenter = np.array([bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] / 2.])


    if crop_height / height > crop_width / width:
        crop_size = crop_height
        min_shape = height

    else:
        crop_size = crop_width
        min_shape = width

    crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
    crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
    crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
    crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)

    min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
    max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
    min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
    max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)

    x_ratio = float(width) / (max_x - min_x)
    y_ratio = float(height) / (max_y - min_y)

    # 256 x 192
    img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (width, height))
    #cv2.imshow('test', img)

    if stage != 'train':
        detail = np.asarray([min_x - add, min_y - add, max_x - add, max_y - add])

    img = img - cfg.pixel_means
    if cfg.pixel_norm:
        img = img / 255.
    img = img.transpose(2, 0, 1)

    imgs.append(img)


    if stage == 'train':
        return 0
    else:
        return [np.asarray(imgs).astype(np.float32), detail]

