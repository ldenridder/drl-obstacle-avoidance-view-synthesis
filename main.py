# --------------------------------------------------------
# Phase Two: Drone Navigation
# Build 2.0.0
# Written by Luc den Ridder
# --------------------------------------------------------

"""
Combining Stereo Image Prediction With Deep RL On Single
Camera Drones
----------------------------------------------------------
Phase Two:
Deep Reinforcement Learning for Monocular Vision-Based
Drones trained with Stereo Vision
----------------------------------------------------------
Build 2.0.0 (14-12-2022)
New features:
- Code Review Ready

"""

""" Load Libraries, Functions, Classes and Configs """
# load libraries
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import wandb
import os

# load functions and classses
from dataset import Kitti, Simulation
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

simulation = {
    'csname': "simulation_config",
    'nodename': "SimulationConfig"}

dataset = simulation
cs = ConfigStore.instance()
cs.store(name=dataset['csname'], node=dataset['nodename'])

""" Main Code """
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: dataset['nodename']):
    # initialise weights&biases and check for GPU
    if dataset['csname'] == "kitti_config":
        dataset['dataloader'] = Kitti
        cfgdir = cfg.kitti

    if dataset['csname'] == "simulation_config":
        dataset['dataloader'] = Simulation
        cfgdir = cfg.simulation

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

    # load the model, optimizer, scheduler and metrics
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model_loader(cfg, cfgdir, device)
    optimizer = build_optimizer(cfgdir, model)
    lr_scheduler = build_scheduler(cfgdir.hyperparams, optimizer, len(train_loader))
    metrics = metrics_loader(device)

    summary(model, [(cfgdir.hyperparams.batch_size, 3, cfgdir.files.image_height+2*cfgdir.files.image_height_pad, cfgdir.files.image_width+2*cfgdir.files.image_width_pad), (1,)], dtypes=[torch.float, torch.int])

    if cfg.settings.wandb==True:
        wandb.watch(model, optimizer, log="all", log_freq=10000)
        
    """Model Training"""
    for epoch in range(cfgdir.hyperparams.epochs):
        for i, (left_images,right_images) in enumerate(train_loader):
            # augmenting the images in batches
            left_images = left_images.float().to(device)
            right_images = right_images.float().to(device)            
            left_images, right_images = data_graft(left_images,right_images)

            """Main Model"""
            # forward pass
            warp_identity = np.random.choice([0,1])
            outputs = output_function(model, left_images, right_images, warp_identity)
            loss, lpips_loss, l1_loss, loss_right, loss_left = loss_function(left_images, right_images, outputs, cfg, warp_identity, device)

            # backward pass
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * len(train_loader) + i))
            loss.backward()
            optimizer.step()

            # logging the pass
            wandb_training_logger(loss, lpips_loss, l1_loss, loss_right, loss_left, epoch, i, cfg)

        """Model Validation and W&B Information"""    
        if epoch % 20 == 0:
            # rendering and storing images and saving the model
            rendering(left_images,right_images,outputs,warp_identity,epoch,cfg,cfgdir,"training")
            save_model(model,run,epoch,cfg,cfgdir)
            print (f'Epoch [{epoch+1}/{cfgdir.hyperparams.epochs}], Loss: {loss.item():.4f}')

            with torch.no_grad():
                losses = {"average": 0, "left":0, "right":0, "SSIM": 0, "L1": 0, "LPIPS": 0}
                metrics_score = {"FID": 0, "SSIM": 0, "PSNR": 0, "MSSIM": 0, "LPIPS": 0}

                for i, (left_images, right_images) in enumerate(val_loader):
                    left_images = left_images.float().to(device)
                    right_images = right_images.float().to(device)

                    # forward pass
                    warp_identity = i % 2
                    outputs = output_function(model, left_images, right_images, warp_identity)
                    metrics_score, losses = loss_function_eval(left_images, right_images, outputs, cfg, cfgdir, device, warp_identity, metrics, metrics_score, losses)
                    
                    wandb.log({"Epoch": epoch})
                    if i == 0:
                        # rendering and storing images
                        rendering(left_images,right_images,outputs,warp_identity,epoch,cfg,cfgdir,"validation")
                        
                # logging the performance
                wandb.log({"Validation (+Test) Loss": losses['average']/len(val_loader), "Validation (+Test) Loss LPIPS": losses['LPIPS']/len(val_loader), "Validation (+Test) Loss L1": losses['L1']/len(val_loader), "Validation (+Test) Loss Right": losses['right']/len(val_loader), "Validation (+Test) Loss Left": losses['left']/len(val_loader),
                    "FID Score": metrics_score["FID"]/len(val_loader), "SSIM Score": metrics_score["SSIM"]/len(val_loader), "PSNR Score": metrics_score["PSNR"]/len(val_loader), "MSSIM Score": metrics_score["MSSIM"]/len(val_loader), "LPIPS Score": metrics_score["LPIPS"]/len(val_loader)})
                
    print('Finished Training')

    """Model Testing"""
    with torch.no_grad():
        losses = {"average": 0, "left":0, "right":0, "SSIM": 0, "L1": 0, "LPIPS": 0}
        metrics_score = {"FID": 0, "SSIM": 0, "PSNR": 0, "MSSIM": 0, "LPIPS": 0}

        for i, (left_images, right_images) in enumerate(test_loader):
            left_images = left_images.float().to(device)
            right_images = right_images.float().to(device)
            
            # forward pass
            warp_identity = i % 2
            outputs = output_function(model, left_images, right_images, warp_identity)
            metrics_score, losses = loss_function_eval(left_images, right_images, outputs, cfg, cfgdir, device, warp_identity, metrics, metrics_score, losses)
            
            if i == 0:
                # rendering and storing images and saving the model
                rendering(left_images,right_images,outputs,warp_identity,epoch,cfg,cfgdir,"testing")
                save_model(model,run,'test',cfg, cfgdir)

        # logging the performance
        wandb.log({"Validation (+Test) Loss": losses['average']/len(test_loader), "Validation (+Test) Loss LPIPS": losses['LPIPS']/len(test_loader), "Validation (+Test) Loss L1": losses['L1']/len(test_loader),
            "FID Score": metrics_score["FID"]/len(test_loader), "SSIM Score": metrics_score["SSIM"]/len(test_loader), "PSNR Score": metrics_score["PSNR"]/len(test_loader), "MSSIM Score": metrics_score["MSSIM"]/len(test_loader), "LPIPS Score": metrics_score["LPIPS"]/len(test_loader)})

if __name__ == '__main__':
    main()