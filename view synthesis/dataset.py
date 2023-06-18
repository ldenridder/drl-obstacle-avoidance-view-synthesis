from logging import root
from pathlib import Path
from typing import Any

import torch
import cv2
import os
import numpy as np
import fnmatch
import skimage.transform
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

from kitti_utils import generate_depth_map

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

class Kitti(Dataset[Any]):
    """Kitti 2015 Stereo dataset"""

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
        idx: int
        if torch.is_tensor(idx):
            idx = idx.totlist()
        
        is_flip = np.random.choice([0,1]) == 1 and self.data_directory == "train"
        is_color_aug = np.random.choice([0,1]) == 1 and self.data_directory == "train"
        color_aug = transforms.ColorJitter(brightness=(0.8, 1.2),
                                                      contrast=(0.8, 1.2),
                                                      saturation=(0.8, 1.2),
                                                      hue=(-0.1, 0.1))

        
        filename = self.filenames[idx].split(" ")
        left_image, right_image = self.get_images(filename)
        left_image, right_image = self.transform(left_image, right_image, is_flip, is_color_aug, color_aug)

        return left_image, right_image

class KittiDepth(Dataset[Any]):
    """Kitti 2015 Depth dataset"""

    def __init__(self, cfg, data_directory):
        self.root_path = cfg.files.path
        self.image_l_path = cfg.files.left_image_data
        self.image_r_path = cfg.files.right_image_data
        self.mean = cfg.hyperparams.mean
        self.std = cfg.hyperparams.std
        self.mean_depth = cfg.hyperparams.mean_depth
        self.std_depth = cfg.hyperparams.std_depth
        self.image_height = cfg.files.image_height
        self.image_width = cfg.files.image_width
        self.image_pad = cfg.files.image_pad
        self.data_directory = data_directory
        self.depth_gt_size = (375, 1242)
        self.garg_crop = False
        self.eigen_crop = True
        self.min_depth = 1e-3
        self.max_depth = 80
        if self.data_directory == "test":
           with open(os.path.join(os.path.dirname(__file__), "splits", "eigen", f"{data_directory}_files.txt"), 'r') as f:    
            self.filenames = np.array(f.read().splitlines()) 
        
        else:
            fpath = os.path.join(os.path.dirname(__file__), "splits", "eigen_full", "{}_files.txt")
            with open(fpath.format(data_directory), 'r') as f:    
                self.filenames = np.array(f.read().splitlines())
        
    def __len__(self):
        return len(self.filenames)

    def eval_mask(self, depth_gt):
        """Following Adabins, Do grag_crop or eigen_crop for testing"""
        depth_gt = np.squeeze(depth_gt)
        valid_mask = np.logical_and(depth_gt > self.min_depth, depth_gt < self.max_depth)
        if self.garg_crop or self.eigen_crop:
            gt_height, gt_width = depth_gt.shape
            eval_mask = np.zeros(valid_mask.shape)

            if self.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif self.eigen_crop:
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
        valid_mask = np.logical_and(valid_mask, eval_mask)
        valid_mask = np.expand_dims(valid_mask, axis=0)
        return valid_mask
    
    def get_images(self, filename):     
        img_name_left = str(Path(f"{self.root_path}/{filename[0]}/{self.image_l_path}/data/{filename[1].zfill(10)}.jpg"))
        left_image = cv2.cvtColor(cv2.imread(img_name_left), cv2.COLOR_BGR2RGB)
        img_name_right = str(Path(f"{self.root_path}/{filename[0]}/{self.image_r_path}/data/{filename[1].zfill(10)}.jpg"))
        right_image = cv2.cvtColor(cv2.imread(img_name_right), cv2.COLOR_BGR2RGB)    
        return left_image, right_image
    
    def get_depth(self,filename,is_flip, cam):
        transform_depth = transforms.Compose([
            transforms.ToTensor()])

        foldername = filename[0].split("/")[0]
        calib_path = str(Path(f"{self.root_path}/{foldername}"))
        velo_filename = str(Path(f"{self.root_path}/{filename[0]}/velodyne_points/data/{filename[1].zfill(10)}.bin"))

        assert os.path.isfile(velo_filename), "Frame {} in {} don't have ground truth".format(filename[1], filename[0])

        depth_gt = generate_depth_map(calib_path, velo_filename, cam, True)
        if is_flip:
            depth_gt = np.fliplr(depth_gt)
            

        #depth_gt = self.eval_kb_crop(depth_gt)
        

        depth_gt = skimage.transform.resize(depth_gt, self.depth_gt_size, order=0, preserve_range=True, mode='constant')
        valid_mask = self.eval_mask(depth_gt)
        depth_gt = transform_depth(depth_gt.copy())

        return depth_gt, valid_mask

    def transform(self, left_image, right_image, is_flip, is_color_aug, color_aug):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(self.image_height,self.image_width)),        
            transforms.Pad((self.image_pad,0)),
            transforms.ToTensor()])     

        transform_hq = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(self.image_height*2,self.image_width*2)),       
            transforms.ToTensor()])     


        if not is_flip:
            return transform(left_image),transform(right_image),transform_hq(left_image),transform_hq(right_image)
        else:
            return transform(right_image),transform(left_image),transform_hq(right_image),transform_hq(left_image)

    def __getitem__(self, idx):
        idx: int
        if torch.is_tensor(idx):
            idx = idx.totlist()
        
        is_flip = np.random.choice([0,1]) == 1 and self.data_directory == "train"
        is_color_aug = np.random.choice([0,1]) == 1 and self.data_directory == "train"
        color_aug = transforms.ColorJitter(brightness=(0.8, 1.2),
                                                      contrast=(0.8, 1.2),
                                                      saturation=(0.8, 1.2),
                                                      hue=(-0.1, 0.1))

        
        filename = self.filenames[idx].split(" ")
        left_image, right_image = self.get_images(filename)
        left_image, right_image, left_image_hq, right_image_hq = self.transform(left_image, right_image, is_flip, is_color_aug, color_aug)
        left_depth_gt, left_valid_mask = self.get_depth(filename, is_flip, 2)
        right_depth_gt, right_valid_mask = self.get_depth(filename, is_flip, 3)

        # left_mask = np.ma.masked_equal(left_depth_gt, 0)

        left_depth = np.squeeze(left_depth_gt)
        # right_depth = np.squeeze(right_depth_gt)

        # left_depth_gt = cv2.inpaint(left_depth_gt.astype(np.float32), left_mask.astype(np.uint8), 3, cv2.INPAINT_NS)
        # vmax_left = np.max(np.array(left_depth.view(352,1216,1)))
        # plt.figure()
        # plt.imshow(left_depth, cmap='magma', vmax=vmax_left)
        # plt.axis('off')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig('left_depth_gt_1.png', bbox_inches='tight',pad_inches= 0)
        # plt.close()

        # vmax_right = np.percentile(right_depth_gt, 99)
        # plt.figure()
        # plt.imshow(right_depth_gt, cmap='magma', vmax=vmax_right)
        # plt.axis('off')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig('right_depth_gt_1.png', bbox_inches='tight',pad_inches= 0)
        # plt.close()
        data = {}
        K = np.array([[0.58, 0, 0.5, 0],
                        [0, 1.92, 0.5, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
        K[0, :] *= self.image_width
        K[1, :] *= self.image_height
        self.K = K
        stereo_T_l = np.eye(4, dtype=np.float32)
        stereo_T_r = stereo_T_l
        baseline_sign = -1 if is_flip else 1
        stereo_T_l[0, 3] = baseline_sign * 0.1
        stereo_T_r[0, 3] = baseline_sign * 0.1 * -1
        data["left image"] = left_image
        data["right image"] = right_image
        data["left image hq"] = left_image_hq
        data["right image hq"] = right_image_hq
        data["left depth gt"] = left_depth_gt
        data["right depth gt"] = right_depth_gt
        data["left valid mask"] = left_valid_mask
        data["right valid mask"] = right_valid_mask
        data["stereo_T-l"] = torch.from_numpy(stereo_T_l)
        data["stereo_T-r"] = torch.from_numpy(stereo_T_r)
        data["K"] = torch.from_numpy(self.K)

        return data

class CityScapes(Dataset[Any]):
    """CityScapes"""

    def __init__(self, root_path: str, data_directory: str, image_l_directory: str, image_r_directory: str, image_height: int, image_height_cropped:int, image_width: int, mean: list, std:list):
        data_path = Path(f"{root_path}/{data_directory}")
        self.image_l_path = Path(f"{root_path}/{image_l_directory}/{data_directory}")
        self.image_r_path = Path(f"{root_path}/{image_r_directory}/{data_directory}")
        self.frame_names_left = []
        for root, dirs, files in os.walk(Path(self.image_l_path)):
            dirs.sort()
            files.sort()
            for i in range(len(files)):
                self.frame_names_left.append(Path(f"{root}/{files[i]}"))
        self.frame_names_right = []
        for root, dirs, files in os.walk(Path(self.image_r_path)):
            dirs.sort()
            files.sort()
            for i in range(len(files)):
                self.frame_names_right.append(Path(f"{root}/{files[i]}"))
        self.mean = mean
        self.std = std
        self.image_height = image_height
        self.image_width = image_width
    
    def __len__(self):
        return len(self.frame_names_left)

    def __getitem__(self, idx):
        idx: int
        if torch.is_tensor(idx):
            idx = idx.totlist()

        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(self.image_height,self.image_width)),
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std)
        ])

        img_name_left = str(self.frame_names_left[idx])
        left_image = cv2.cvtColor(cv2.imread(img_name_left), cv2.COLOR_BGR2RGB)
        left_image = transform(left_image)

        img_name_right = str(self.frame_names_right[idx])
        right_image = cv2.cvtColor(cv2.imread(img_name_right), cv2.COLOR_BGR2RGB)
        right_image = transform(right_image)    

        return left_image, right_image


""" Dataset Classes """
class Kitti_Tulsiani(Dataset[Any]):
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
        self.filenames = []
        self.seq_id_list = []
       
        exclude_img = '2011_09_26_drive_0117_sync/image_02/data/0000000074.jpg'
        seq_names = self.raw_city_sequences()
        rng = np.random.RandomState(0)
        rng.shuffle(seq_names)
        n_all = len(seq_names)
        n_train = int(round(0.7 * n_all))
        n_val = int(round(0.15 * n_all))
        if data_directory == "train":
            seq_names = seq_names[0:n_train]
        if data_directory == "validation":
            seq_names = seq_names[n_train:(n_train + n_val)]
        if data_directory == "test":
            seq_names = seq_names[(n_train + n_val):n_all]
        for seq_id in seq_names:
            seq_date = seq_id[0:10]
            seq_dir = os.path.join(self.root_path, seq_date,
                                '{}_sync'.format(seq_id))
            for root, _, filenames in os.walk(os.path.join(seq_dir, 'image_02')):
                for filename in fnmatch.filter(filenames, '*.jpg'):
                    src_img_name = os.path.join(root, filename)
                    if exclude_img not in src_img_name:
                        self.filenames.append(src_img_name[-14:-4])
                        self.seq_id_list.append(f"{seq_date}/{seq_id}_sync")
        # print(len(self.filenames))

    def __len__(self):
        return len(self.filenames)

    def raw_city_sequences(self):
        """Sequence names for city sequences in kitti raw data.

        Returns:
            seq_names: list of names
        """
        seq_names = [
            '2011_09_26_drive_0001',
            '2011_09_26_drive_0002',
            '2011_09_26_drive_0005',
            '2011_09_26_drive_0009',
            '2011_09_26_drive_0011',
            '2011_09_26_drive_0013',
            '2011_09_26_drive_0014',
            '2011_09_26_drive_0017',
            '2011_09_26_drive_0018',
            '2011_09_26_drive_0048',
            '2011_09_26_drive_0051',
            '2011_09_26_drive_0056',
            '2011_09_26_drive_0057',
            '2011_09_26_drive_0059',
            '2011_09_26_drive_0060',
            '2011_09_26_drive_0084',
            '2011_09_26_drive_0091',
            '2011_09_26_drive_0093',
            '2011_09_26_drive_0095',
            '2011_09_26_drive_0096',
            '2011_09_26_drive_0104',
            '2011_09_26_drive_0106',
            '2011_09_26_drive_0113',
            '2011_09_26_drive_0117',
            '2011_09_28_drive_0001',
            '2011_09_28_drive_0002',
            '2011_09_29_drive_0026',
            '2011_09_29_drive_0071',
        ]
        return seq_names


    def get_images(self, filename, seq_id):     
        img_name_left = str(Path(f"{self.root_path}/{seq_id}/{self.image_l_path}/data/{filename}.jpg"))
        left_image = cv2.cvtColor(cv2.imread(img_name_left), cv2.COLOR_BGR2RGB)
        img_name_right = str(Path(f"{self.root_path}/{seq_id}/{self.image_r_path}/data/{filename}.jpg"))
        right_image = cv2.cvtColor(cv2.imread(img_name_right), cv2.COLOR_BGR2RGB)    
        return left_image, right_image

    def transform(self, left_image, right_image, is_flip, color_aug, is_color_aug):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(self.image_height,self.image_width-2*self.image_pad)),
            RandomFlipImage(is_flip=is_flip),
            RandomColorAugImage(is_color_aug,color_aug),           
            transforms.Pad((self.image_pad,0)),
            transforms.ToTensor()])       

        if not is_flip:
            return transform(left_image),transform(right_image)
        else:
            return transform(right_image),transform(left_image)

    def __getitem__(self, idx):
        idx: int
        if torch.is_tensor(idx):
            idx = idx.totlist()
        
        is_flip = np.random.choice([0,1]) == 1 and self.data_directory == "train"
        is_color_aug = np.random.choice([0,1]) == 1 and self.data_directory == "train"
        color_aug = transforms.ColorJitter(brightness=(0.8, 1.2),
                                                      contrast=(0.8, 1.2),
                                                      saturation=(0.8, 1.2),
                                                      hue=(-0.1, 0.1))


        left_image, right_image = self.get_images(self.filenames[idx], self.seq_id_list[idx])
        left_image, right_image = self.transform(left_image, right_image, is_flip, color_aug, is_color_aug)

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

        for root, dirs, files in os.walk(Path(f"{self.root_path}/{self.data_directory}/left")):
            dirs.sort()
            files.sort()

            self.filenames = files    
        
    def __len__(self):
        return len(self.filenames)

    def get_images(self, filename):     
        img_name_left = str(Path(f"{self.root_path}/{self.data_directory}/{self.image_l}/{filename}"))
        left_image = cv2.cvtColor(cv2.imread(img_name_left), cv2.COLOR_BGR2RGB)

        img_name_right = str(Path(f"{self.root_path}/{self.data_directory}/{self.image_r}/{filename}"))
        right_image = cv2.cvtColor(cv2.imread(img_name_right), cv2.COLOR_BGR2RGB)
        # right_image = cv2.imread(img_name_right)
        return left_image, right_image

    def transform(self, image:np.ndarray) -> torch.Tensor:
        transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor()]) 
        return transform(image)

    def transform_depth(self, image:np.ndarray) -> torch.Tensor:

        transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor()]) 
        return transform(image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.totlist()
        

        filename = self.filenames[idx]
        left_image, right_image = self.get_images(filename)
        left_image = self.transform(left_image)
        right_image = self.transform_depth(right_image)
        return left_image, right_image