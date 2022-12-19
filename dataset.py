# --------------------------------------------------------
# Phase Two: Drone Navigation
# Build 2.0.0
# Written by Luc den Ridder
# --------------------------------------------------------

""" Load Libraries, Functions, Classes """
from pathlib import Path
from typing import Any

import torch
import cv2
import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

""" Image Augmentation Classes """
class RandomFlipImage(object):
    def __init__(self, is_flip):
        self.is_flip = is_flip

    def __call__(self, x):
        if not self.is_flip:
            return x
        out = x.transpose(Image.FLIP_LEFT_RIGHT)
        return out

class RandomColorAugImage(object):
    def __init__(self, is_color_aug, color_aug):
        self.color_aug = color_aug
        self.is_color_aug = is_color_aug

    def __call__(self, x):
        if not self.is_color_aug:
            return x
        out = self.color_aug(x)
        return out


""" Dataset Classes """
class Kitti(Dataset[Any]):
    """ Kitti 2015 Stereo dataset """

    def __init__(self, cfg, data_directory):
        self.root_path = cfg.files.path
        self.image_l_path = cfg.files.left_image_data
        self.image_r_path = cfg.files.right_image_data
        self.mean = cfg.hyperparams.mean
        self.std = cfg.hyperparams.std
        self.image_height = cfg.files.image_height
        self.image_width = cfg.files.image_width
        self.image_pad = cfg.files.image_pad
        self.data_directory = data_directory
        if data_directory == "test":
           with open(os.path.join(os.path.dirname(__file__), "splits", "eigen", f"{data_directory}_files.txt"), 'r') as f:    
            self.filenames = np.array(f.read().splitlines()) 
        
        else:
            fpath = os.path.join(os.path.dirname(__file__), "splits", "eigen_full", "{}_files.txt")
            with open(fpath.format(data_directory), 'r') as f:    
                self.filenames = np.array(f.read().splitlines())
        
    def __len__(self):
        return len(self.filenames)

    def get_images(self, filename):     
        img_name_left = str(Path(f"{self.root_path}/{filename[0]}/{self.image_l_path}/data/{filename[1].zfill(10)}.jpg"))
        left_image = cv2.cvtColor(cv2.imread(img_name_left), cv2.COLOR_BGR2RGB)
        img_name_right = str(Path(f"{self.root_path}/{filename[0]}/{self.image_r_path}/data/{filename[1].zfill(10)}.jpg"))
        right_image = cv2.cvtColor(cv2.imread(img_name_right), cv2.COLOR_BGR2RGB)    
        return left_image, right_image

    def transform(self, left_image, right_image, is_flip, is_color_aug, color_aug):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(self.image_height,self.image_width)),
            RandomFlipImage(is_flip=is_flip),
            RandomColorAugImage(is_color_aug=is_color_aug, color_aug=color_aug),           
            transforms.Pad((self.image_pad,0)),
            transforms.ToTensor()])     

        if not is_flip:
            return transform(left_image),transform(right_image)
        else:
            return transform(right_image),transform(left_image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.totlist()
        
        is_flip = np.random.choice([0,1]) == 1 and self.data_directory == "train"
        is_color_aug = np.random.choice([0,1]) == 1 and self.data_directory == "train"
        color_aug = transforms.ColorJitter.get_params(brightness=(0.8, 1.2),
                                                      contrast=(0.8, 1.2),
                                                      saturation=(0.8, 1.2),
                                                      hue=(-0.1, 0.1))
        
        filename = self.filenames[idx].split(" ")
        left_image, right_image = self.get_images(filename)
        left_image, right_image = self.transform(left_image, right_image, is_flip, is_color_aug, color_aug)

        return left_image, right_image

class Simulation(Dataset[Any]):
    """ Synthetic Avoidbench dataset """

    def __init__(self, cfg, data_directory):
        self.root_path = cfg.files.path
        self.image_l = cfg.files.left_image_data
        self.image_r = cfg.files.right_image_data
        self.image_height = cfg.files.image_height
        self.image_width = cfg.files.image_width
        self.image_height_pad = cfg.files.image_height_pad
        self.image_width_pad = cfg.files.image_width_pad
        self.data_directory = data_directory
        self.filenames = []

        for root, dirs, files in os.walk(Path(f"{self.root_path}/{self.data_directory}")):
            dirs.sort()
            files.sort()
            for i in range(0,len(files),2):
                self.filenames.append(f"{files[i]}")
        
    def __len__(self):
        return len(self.filenames)

    def get_images(self, filename):     
        img_name_left = str(Path(f"{self.root_path}/{self.data_directory}/{self.image_l}/{filename}"))
        left_image = cv2.cvtColor(cv2.imread(img_name_left), cv2.COLOR_BGR2RGB)

        img_name_right = str(Path(f"{self.root_path}/{self.data_directory}/{self.image_r}/{filename}"))
        right_image = cv2.cvtColor(cv2.imread(img_name_right), cv2.COLOR_BGR2RGB)    
        return left_image, right_image

    def transform(self, left_image, right_image, is_flip, is_color_aug, color_aug):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(self.image_height,self.image_width)),
            RandomFlipImage(is_flip=is_flip),
            RandomColorAugImage(is_color_aug=is_color_aug, color_aug=color_aug),           
            transforms.Pad((self.image_width_pad,self.image_height_pad)),
            transforms.ToTensor()])     

        if not is_flip:
            return transform(left_image),transform(right_image)
        else:
            return transform(right_image),transform(left_image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.totlist()
        
        is_flip = np.random.choice([0,1]) == 1 and self.data_directory == "Indoor"
        is_color_aug = np.random.choice([0,1]) == 1 and self.data_directory == "Indoor"
        color_aug = transforms.ColorJitter.get_params(brightness=(0.8, 1.2),
                                                      contrast=(0.8, 1.2),
                                                      saturation=(0.8, 1.2),
                                                      hue=(-0.1, 0.1))

        filename = self.filenames[idx]
        left_image, right_image = self.get_images(filename)
        left_image, right_image = self.transform(left_image, right_image, is_flip, is_color_aug, color_aug)
        return left_image, right_image