# --------------------------------------------------------
# Step 01: Autoencoder
# Build 1.1.0
# Written by Luc den Ridder
# --------------------------------------------------------

"""
Geometry-free mono-to-stereo image rendering as an
auxiliary task to reinforcmeent learning.
----------------------------------------------------------
Step 01: Autoencoder
For this task an autoencoder based on the SWIN
architecture is developed that can predict stereo images
from a single image.
----------------------------------------------------------
Build 1.2.0 (23-09-2022)
New features:
- SWIN is implemented

Bug fix:


Future ideas:
- documenting all code
- github
"""

""" Load Libraries, Functions, Classes and Configs """
# load libraries
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import wandb
import os

# load functions and classses
from dataset import Kitti, CityScapes, KittiDepth
from config import KITTIConfig
from models import LossEncoder, SwinTransformer, SwinTransformerDepth
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from utils import *


# select gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

# load config
kitti = {
    'csname': "kitti_config",
    'nodename': "KITTIConfig"}

cityscapes = {
    'csname': "cityscapes_config",
    'nodename': "CityScapesConfig"}

dataset = kitti

cs = ConfigStore.instance()
cs.store(name=dataset['csname'], node=dataset['nodename'])


""" Main Code """
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: dataset['nodename']):
    # initialise weights&biases and check for GPU
    if dataset['csname'] == "kitti_config":
        dataset['dataloader'] = Kitti
        cfgdir = cfg.kitti
    if dataset['csname'] == "cityscapes_config":
        dataset['dataloader'] = CityScapes
        cfgdir = cfg.cityscapes

    if cfg.settings.wandb==True:
        run = wandb_loader(cfg,cfgdir,prefix="Test: ")
        artifact = run.use_artifact(f'ldenridder/Autoencoder (1.1.2)/SWINKITTI_lr-{cfgdir.hyperparams.base_lr}_bs-{cfgdir.hyperparams.batch_size}_epoch-test:v6', type='model')
        artifact.download()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loading in the dataset
    test_dataset = dataset['dataloader'](cfgdir, cfgdir.files.test_data)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfgdir.hyperparams.batch_size, shuffle=True)

    n_layer = 5
    # load model, loss function and optimizer
    model = model_loader(cfg, cfgdir, device)
    model_all = model_loader_test(cfg, cfgdir, device,8)
    model_1 = model_loader_test(cfg, cfgdir, device,9)
    model_2 = model_loader_test(cfg, cfgdir, device,10)
    model_3 = model_loader_test(cfg, cfgdir, device,11)

    checkpoint = torch.load(f'artifacts/SWINKITTI_lr-{cfgdir.hyperparams.base_lr}_bs-{cfgdir.hyperparams.batch_size}_epoch-test:v6/tmp_model.pth')
    model.load_state_dict(checkpoint, strict=False)
    model_1.load_state_dict(checkpoint, strict=False)
    model_2.load_state_dict(checkpoint, strict=False)
    # model_3.load_state_dict(checkpoint, strict=False)
    model_all.load_state_dict(checkpoint, strict=False)

    criterion = eval(f"torch.nn.{cfg.settings.loss_functions[cfg.settings.loss_option]}()")
    metrics = metrics_loader(device)
  
    summary(model, [(cfgdir.hyperparams.batch_size, 3, cfgdir.files.image_height,  cfgdir.files.image_width+2*cfgdir.files.image_pad), (1,)], dtypes=[torch.float, torch.int])
    
    if cfg.settings.wandb==True:
        wandb.watch(model, criterion, log="all", log_freq=100)

    """Model Testing"""
    with torch.no_grad():
        losses = {"average": 0, "left":0, "right":0, "SSIM": 0, "L1": 0, "LPIPS": 0}
        metrics_score = {"FID": 0, "SSIM": 0, "PSNR": 0, "MSSIM": 0, "LPIPS": 0}
        total_classes = []
        total_res_pred = []
        for i, (left_images, right_images) in enumerate(test_loader):
            """Futer fix"""
            if i == len(test_loader)-1:
                break
            if i == 1:
                break
            
            left_images = left_images.float().to(device)
            right_images = right_images.float().to(device)
            # both_images = torch.stack((left_images,right_images), dim=1)
            right_images_prediction = []
            left_images_prediction = []
            warp_identity = i % 2
            #outputs = output_function(model, left_images, right_images, warp_identity)
            for mod in [model, model_1, model_2, model_3, model_all]:
                right_images_prediction.append(output_function(mod, left_images, right_images, 0))
                left_images_prediction.append(output_function(mod, left_images, right_images, 1))
                # metrics_score, losses = loss_function_eval(left_images, right_images, outputs, cfg, cfgdir, device, warp_identity, metrics, metrics_score, losses)
            # right_images_prediction.append(output_function(model, left_images, right_images, 0))
            # right_images_prediction.append(output_function(model, right_images, left_images, 0))
            # left_images_prediction.append(output_function(model, left_images, right_images, 1))
            # left_images_prediction.append(output_function(model, right_images, left_images, 1))
            # if i < 10:
            #     # rendering_test(left_images,right_images,outputs,n_layer,i,cfg,cfgdir,"testing")
            #     total_classes, total_res_pred = get_testing_batches(outputs,n_layer,cfg,cfgdir,total_classes,total_res_pred)
            # else:
            #     rendering_testing_scatter(total_classes,total_res_pred,i,cfg)
            #     #rendering_scatter(outputs,n_layer,i,cfg,cfgdir,"test")
            if i == 0:
                rendering_disparities_layered(left_images, right_images, left_images_prediction, right_images_prediction, cfgdir)

        wandb.log({"Validation (+Test) Loss": losses['average']/len(test_loader), "Validation (+Test) Loss LPIPS": losses['LPIPS']/len(test_loader), "Validation (+Test) Loss L1": losses['L1']/len(test_loader),
            "FID Score": metrics_score["FID"]/len(test_loader), "SSIM Score": metrics_score["SSIM"]/len(test_loader), "PSNR Score": metrics_score["PSNR"]/len(test_loader), "MSSIM Score": metrics_score["MSSIM"]/len(test_loader), "LPIPS Score": metrics_score["LPIPS"]/len(test_loader)})


if __name__ == '__main__':
    main()