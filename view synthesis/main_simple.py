# --------------------------------------------------------
# Step 01: Autoencoder
# Build 1.1.2
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
Build 1.1.2 (29-09-2022)
New features:
- Kitti split compatible
- Validation after every 5 epochs

Bug fix:


Future ideas:
- documenting all code
- github
"""

""" Load Libraries, Functions, Classes and Configs """
# load libraries
from turtle import update
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import wandb
import os

# load functions and classses
from dataset import CityScapes, Kitti
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
        run = wandb_loader(cfg,cfgdir)
        
    # loading in the dataset
    train_dataset = dataset['dataloader'](cfgdir, cfgdir.files.train_data)
    val_dataset = dataset['dataloader'](cfgdir, cfgdir.files.val_data)
    test_dataset = dataset['dataloader'](cfgdir, cfgdir.files.test_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfgdir.hyperparams.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=cfgdir.hyperparams.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfgdir.hyperparams.batch_size, shuffle=False)

    # mean_std_printer(train_loader)

    # load model, loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model = model_loader(cfg, cfgdir, device)
    criterion = eval(f"torch.nn.{cfg.settings.loss_functions[cfg.settings.loss_option]}()")
    optimizer = build_optimizer(cfgdir, model)
    lr_scheduler = build_scheduler(cfgdir.hyperparams, optimizer, len(train_loader))
    metrics = metrics_loader(device)

    summary(model, (cfgdir.hyperparams.batch_size, 3, cfgdir.files.image_height, cfgdir.files.image_width+2*cfgdir.files.image_pad))

    if cfg.settings.wandb==True:
        wandb.watch(model, optimizer, log="all", log_freq=10000)
        
    """Model Training"""
    for epoch in range(cfgdir.hyperparams.epochs):
        for i, (left_images,right_images) in enumerate(train_loader):
            left_images = left_images.float().to(device)
            right_images = right_images.float().to(device)            
            left_images, right_images = data_graft(left_images,right_images)
            """Main Model"""
            # Forward pass
            outputs = model(left_images)
            l1 = torch.nn.L1Loss()
            loss = l1(outputs,right_images)

            # Backward and optimize
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * len(train_loader) + i))
            loss.backward()
            optimizer.step()

            if cfg.settings.wandb==True and i % 10 == 0:
                wandb.log({"Epoch": epoch, "Train Loss": loss})

        """W&B Information"""    
        if epoch % 5 == 0:
            rendering(left_images,right_images,outputs,0,epoch,cfg,cfgdir,"training")
            save_model(model,run,epoch,cfg,cfgdir)
            print (f'Epoch [{epoch+1}/{cfgdir.hyperparams.epochs}], Loss: {loss.item():.4f}')

            with torch.no_grad():
                losses = {"average": 0, "left":0, "right":0, "SSIM": 0, "L1": 0, "LPIPS": 0}
                metrics_score = {"FID": 0, "SSIM": 0, "PSNR": 0, "MSSIM": 0, "LPIPS": 0}

                for i, (left_images, right_images) in enumerate(val_loader):
                    left_images = left_images.float().to(device)
                    right_images = right_images.float().to(device)
                    # both_images = torch.stack((left_images,right_images), dim=1)
                    outputs = model(left_images)
                    metrics_score = metrics_compute(right_images, outputs, metrics, metrics_score, cfgdir)

                    wandb.log({"Epoch": epoch})
                    if i == 0:
                        rendering(left_images,right_images,outputs,0,epoch,cfg,cfgdir,"validation")
                        
                wandb.log({"Validation (+Test) Loss": losses['average']/len(val_loader), "Validation (+Test) Loss LPIPS": losses['LPIPS']/len(val_loader), "Validation (+Test) Loss L1": losses['L1']/len(val_loader), "Validation (+Test) Loss Right": losses['right']/len(val_loader), "Validation (+Test) Loss Left": losses['left']/len(val_loader),
                    "FID Score": metrics_score["FID"]/len(val_loader), "SSIM Score": metrics_score["SSIM"]/len(val_loader), "PSNR Score": metrics_score["PSNR"]/len(val_loader), "MSSIM Score": metrics_score["MSSIM"]/len(val_loader), "LPIPS Score": metrics_score["LPIPS"]/len(val_loader)})
                
    print('Finished Training')

    """Model Testing"""
    with torch.no_grad():
        losses = {"average": 0, "left":0, "right":0, "SSIM": 0, "L1": 0, "LPIPS": 0}
        metrics_score = {"FID": 0, "SSIM": 0, "PSNR": 0, "MSSIM": 0, "LPIPS": 0}

        for i, (left_images, right_images) in enumerate(test_loader):
            """Futer fix"""
            if i == len(test_loader)-1:
                break
            
            left_images = left_images.float().to(device)
            right_images = right_images.float().to(device)
            # both_images = torch.stack((left_images,right_images), dim=1)
            
            warp_identity = i % 2
            outputs = model(left_images)
            metrics_score = metrics_compute(right_images, outputs, metrics, metrics_score, cfgdir)

            if i == 0:
                rendering(left_images,right_images,outputs,warp_identity,epoch,cfg,cfgdir,"testing")
                save_model(model,run,'test',cfg, cfgdir)

        wandb.log({"Validation (+Test) Loss": losses['average']/len(test_loader), "Validation (+Test) Loss LPIPS": losses['LPIPS']/len(test_loader), "Validation (+Test) Loss L1": losses['L1']/len(test_loader),
            "FID Score": metrics_score["FID"]/len(test_loader), "SSIM Score": metrics_score["SSIM"]/len(test_loader), "PSNR Score": metrics_score["PSNR"]/len(test_loader), "MSSIM Score": metrics_score["MSSIM"]/len(test_loader), "LPIPS Score": metrics_score["LPIPS"]/len(test_loader)})

if __name__ == '__main__':
    main()