import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import copy
import random
from PIL import ImageOps, ImageEnhance

def img_aug_identity(img, scale=None):
    return img

def img_aug_autocontrast(img, scale=None):
    return ImageOps.autocontrast(img)

def img_aug_equalize(img, scale=None):
    return ImageOps.equalize(img)

def img_aug_contrast(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Contrast(img).enhance(v)

def img_aug_brightness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Brightness(img).enhance(v)

def img_aug_color(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Color(img).enhance(v)

def img_aug_sharpness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Sharpness(img).enhance(v)

def img_aug_hue(img, scale=[0, 0.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v += min_v
    if random.random() < 0.5:
        hue_factor = -v
    else:
        hue_factor = v

    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img

    h, s, v = img.convert("HSV").split()
    np_h = np.array(h, dtype=np.uint8)

    shift = int(round(hue_factor * 255))
    np_h = (np_h.astype(np.int16) + shift) % 256
    np_h = np_h.astype(np.uint8)

    h = Image.fromarray(np_h, "L")
    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img

def img_aug_posterize(img, scale=[4, 8]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    return ImageOps.posterize(img, v)

def img_aug_solarize(img, scale=[1, 256]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    return ImageOps.solarize(img, v)

def get_augment_list():
    return [
        (img_aug_identity, None),
        (img_aug_autocontrast, None),
        (img_aug_equalize, None),
        (img_aug_contrast, [0.05, 0.95]),
        (img_aug_brightness, [0.05, 0.95]),
        (img_aug_color, [0.05, 0.95]),
        (img_aug_sharpness, [0.05, 0.95]),
        (img_aug_posterize, [4, 8]),
        (img_aug_solarize, [1, 256]),
        (img_aug_hue, [0, 0.5]),
    ]

class strong_img_aug:
    def __init__(self, num_augs=6, flag_using_random_num=True):
        assert 1 <= num_augs <= 10
        self.n = num_augs
        self.augment_list = get_augment_list()
        self.flag_using_random_num = flag_using_random_num

    def __call__(self, img):
        if self.flag_using_random_num:
            max_num = np.random.randint(1, self.n + 1)
        else:
            max_num = self.n

        ops = random.choices(self.augment_list, k=max_num)
        for op, scales in ops:
            img = op(img, scales)
        return img

class FundusSeg_Loader(Dataset):
    def __init__(self, data_path, is_train, dataset_name, data_mean, data_std):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.data_mean = data_mean
        self.data_std = data_std

        if self.dataset_name == "drive" or self.dataset_name == "chase" or self.dataset_name == "rc-slo":
            self.imgs_path = glob.glob(os.path.join(data_path, 'img/*.tif'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))
        if self.dataset_name == "stare":
            self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.tif'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))
        if self.dataset_name == "rimone" or self.dataset_name == "refuge" or self.dataset_name == "refuge2":
            self.imgs_path = glob.glob(os.path.join(data_path, 'img/*.jpg'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))

        self.is_train = is_train
        self.augseg1 = strong_img_aug(num_augs = 6, flag_using_random_num = True)

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        if self.dataset_name == "drive":
            label_path = image_path.replace('img', 'label')
            if self.is_train == 1:
                label_path = label_path.replace('_training.tif', '_manual1.tif') 
            else:
                label_path = label_path.replace('_test.tif', '_manual1.tif') 

        if self.dataset_name == "chase":
            label_path = image_path.replace('img', 'label')
            label_path = label_path.replace('.tif', '_1stHO.tif') 

        if self.dataset_name == "stare":
            label_path = image_path.replace('image', 'label')

        if self.dataset_name == "rc-slo":
            label_path = image_path.replace('img', 'label')

        if self.dataset_name == "rimone" or self.dataset_name == "refuge" or self.dataset_name == "refuge2":
            label_path = image_path.replace('img', 'label')
            label_path = label_path.replace('.jpg', '.tif') 

        image = Image.open(image_path)
        label = Image.open(label_path)
        label = label.convert('L')
        raw_height = image.size[1]
        raw_width = image.size[0]

        if self.dataset_name == "drive":
            image, label = self.padding_image(image, label, 594, 594)
        if self.dataset_name == "stare":
            image, label = self.padding_image(image, label, 702, 702)
        if self.dataset_name == "rimone" or self.dataset_name == "refuge" or self.dataset_name == "refuge2":
            image = image.resize((256,256))
            label = label.resize((256,256))

        # Online augmentation
        if self.is_train == 1:
            if torch.rand(1).item() <= 0.5:
                image, label = self.randomRotation(image, label)

            if torch.rand(1).item() <= 0.25:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

            if torch.rand(1).item() <= 0.25:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)

        # ===== 对应原文 AbdominalDataset.__getitem__() =====
        # 原文逻辑：
        #   img1 = self.augseg1(img)
        #   img2 = img
        if self.is_train == 1:
            base_img = image.copy()
            img1 = self.augseg1(base_img)   # 原文 img1
            img2 = base_img                 # 原文 img2
        else:
            img1 = image
            img2 = image

        img1 = np.asarray(img1)
        img2 = np.asarray(img2)
        label = np.asarray(label)

        label = label.reshape(1, label.shape[0], label.shape[1])
        label = np.array(label)

        if (label.max() == 255):
            label = label / 255

        # Normalize img1 / img2 separately
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        img1 = img1 / 127.5
        img1 = img1 - 1.0

        img2 = img2 / 127.5
        img2 = img2 - 1.0

        img1 = img1.transpose(2, 0, 1)
        img2 = img2.transpose(2, 0, 1)

        sp = image_path.split('/')
        filename = sp[len(sp)-1]
        filename = filename[0:len(filename)-4] # del .tif

        if self.is_train == 1:
            return img1, img2, label, filename, raw_height, raw_width
        else:
            return img2, label, filename, raw_height, raw_width

    def __len__(self):
        return len(self.imgs_path)

    def randomRotation(self, image, label, mode=Image.BICUBIC):
        random_angle = torch.randint(low=0,high=360,size=(1,1)).long().item()
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)

    def padding_image(self,image, label, pad_to_h, pad_to_w):
        new_image = Image.new('RGB', (pad_to_w, pad_to_h), (0, 0, 0))
        new_label = Image.new('P', (pad_to_w, pad_to_h), (0, 0, 0))
        new_image.paste(image, (0, 0))
        new_label.paste(label, (0, 0))
        return new_image, new_label