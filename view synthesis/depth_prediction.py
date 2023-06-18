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
        dataset['dataloader'] = KittiDepth
        cfgdir = cfg.kitti
    if dataset['csname'] == "cityscapes_config":
        dataset['dataloader'] = CityScapes
        cfgdir = cfg.cityscapes

    if cfg.settings.wandb==True:
        run = wandb_loader(cfg,cfgdir,prefix="Depth: ")
        artifact = run.use_artifact(f'ldenridder/Autoencoder (1.1.2)/SWIN{cfgdir.files.dataloader}_lr-{cfgdir.hyperparams.base_lr}_bs-{cfgdir.hyperparams.batch_size}_epoch-test:v5', type='model')
        artifact.download()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loading in the dataset
    train_dataset = dataset['dataloader'](cfgdir, cfgdir.files.train_data)  
    val_dataset = dataset['dataloader'](cfgdir, cfgdir.files.train_data)
    test_dataset = dataset['dataloader'](cfgdir, cfgdir.files.train_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfgdir.hyperparams.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=cfgdir.hyperparams.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfgdir.hyperparams.batch_size, shuffle=True)

    # mean_std_depth_printer(train_loader)

    # load model, loss function and optimizer
    model = model_loader_depth(cfg,cfgdir, device)
    checkpoint = torch.load(f'artifacts/SWIN{cfgdir.files.dataloader}_lr-{cfgdir.hyperparams.base_lr}_bs-{cfgdir.hyperparams.batch_size}_epoch-test:v5/tmp_model.pth')
    model.load_state_dict(checkpoint, strict=False)
    if cfg.settings.models[cfg.settings.model_option] == "SWIN":
        for name, param in model.named_parameters():
            if name != 'module.depth_output.weight' and 'up' not in name and 'concat_back_dim' not in name:
                param.requires_grad = False
    elif cfg.settings.models[cfg.settings.model_option] == "UNet":
        for name, param in model.named_parameters():     
            if 'ConvDepth' not in name and 'Up' not in name:
                param.requires_grad = False

    criterion = eval(f"torch.nn.{cfg.settings.loss_functions[cfg.settings.loss_option]}()")
    optimizer = build_optimizer(cfgdir, model)
    lr_scheduler = build_scheduler(cfgdir.hyperparams, optimizer, len(train_loader))
    
    summary(model, (cfgdir.hyperparams.batch_size, 3, cfgdir.files.image_height, cfgdir.files.image_width_padded))
    
    if cfg.settings.wandb==True:
        wandb.watch(model, criterion, log="all", log_freq=100)

    """Model Training"""
    for epoch in range(cfgdir.hyperparams.epochs):
        for i, (data) in enumerate(train_loader):
            for n in data:       
                if n == "left valid mask" or n == "right valid mask":
                    data[n] = data[n].to(device)    
                else:
                    data[n] = data[n].float().to(device)

            """Main Model"""
            # Forward pass
            warp_identity = np.random.choice([0,1])
            outputs, depth_gt, depth_outputs, visual_depth_gt, visual_depth_outputs = output_function_depthv2_eval(model, data, warp_identity, cfgdir, device)
            loss, lpips_loss, l1_loss, loss_right, loss_left = loss_function_depth(data, outputs, cfg, warp_identity, device)


            # Backward and optimize
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * len(train_loader) + i))
            loss.backward()
            optimizer.step()

            if cfg.settings.wandb==True and i % 10 == 0:
                wandb.log({"Epoch": epoch, "Train Loss": loss})

        """W&B Information"""    
        if epoch % 5 == 0:
            rendering_depth(data['left image'],data['right image'],outputs,visual_depth_gt,visual_depth_outputs,warp_identity,epoch,cfg,cfgdir,"training")
            save_model_depth(model,run,epoch,cfg,cfgdir)
            print (f'Epoch [{epoch+1}/{cfgdir.hyperparams.epochs}], Loss: {loss.item():.4f}')
          
            with torch.no_grad():
                total_loss = 0
                errors = {"a1": 0, "a2": 0, "a3": 0, "rmse": 0, "rmse log": 0, "abs rel": 0, "sq rel": 0}
                for i, (data) in enumerate(val_loader):
                    for n in data:       
                        if n == "left valid mask" or n == "right valid mask":
                            data[n] = data[n].to(device)    
                        else:
                            data[n] = data[n].float().to(device)

                    """Main Model"""
                    warp_identity = np.random.choice([0,1])
                    outputs, depth_gt, depth_outputs, visual_depth_gt, visual_depth_outputs = output_function_depthv2_eval(model, data, warp_identity, cfgdir, device)

                    loss, lpips_loss, l1_loss, loss_right, loss_left = loss_function_depth(data, outputs, cfg, warp_identity, device)


                    total_loss += loss
                    
                    if cfg.settings.wandb==True and i % 10 == 0:
                        wandb.log({"Epoch": epoch})
                    if i==0 and cfg.settings.rendering==True:    
                        rendering_depth(data['left image'],data['right image'],outputs,visual_depth_gt,visual_depth_outputs,warp_identity,epoch,cfg,cfgdir,"validation")
                    error = compute_depth_errors(depth_gt, depth_outputs)
                    for item in errors:
                        errors[item] += error[item]

                average_loss = total_loss/len(val_loader)
                
                wandb.log({"Validation (+Test) Loss Average": average_loss, "a1": errors["a1"]/len(val_loader), "a2": errors["a2"]/len(val_loader), "a3": errors["a3"]/len(val_loader), "rmse": errors["rmse"]/len(val_loader), "rmse log": errors["rmse log"]/len(val_loader), "abs rel": errors["abs rel"]/len(val_loader), "sq rel": errors["sq rel"]/len(val_loader)})

    print('Finished Training')

    """Model Testing"""
    with torch.no_grad():
        total_loss = 0
        errors = {"a1": 0, "a2": 0, "a3": 0, "rmse": 0, "rmse log": 0, "abs rel": 0, "sq rel": 0}
        """Futer fix"""
        for i, (data) in enumerate(test_loader):
            if i == len(test_loader)-1:
                break
            for n in data:   
                if n == "left valid mask" or n == "right valid mask":
                    data[n] = data[n].to(device)    
                else:
                    data[n] = data[n].float().to(device)
            """Main Model"""
            warp_identity = np.random.choice([0,1])
            outputs, depth_gt, depth_outputs, visual_depth_gt, visual_depth_outputs = output_function_depthv2_eval(model, data, warp_identity, cfgdir, device)

            loss, lpips_loss, l1_loss, loss_right, loss_left = loss_function_depth(data, outputs, cfg, warp_identity, device)
            total_loss += loss
            
            if cfg.settings.wandb==True and i % 10 == 0:
                wandb.log({"Epoch": epoch})
            if i==0 and cfg.settings.rendering==True:    
                rendering_depth(data['left image'],data['right image'],outputs,visual_depth_gt,visual_depth_outputs,warp_identity,epoch,cfg,cfgdir,"testing")
                save_model_depth(model,run,'test',cfg,cfgdir)
            error = compute_depth_errors(depth_gt, depth_outputs)
            for item in errors:
                errors[item] += error[item]

        average_loss = total_loss/len(test_loader)
        wandb.log({"Validation (+Test) Loss Average": average_loss, "a1": errors["a1"]/len(test_loader), "a2": errors["a2"]/len(test_loader), "a3": errors["a3"]/len(test_loader), "rmse": errors["rmse"]/len(test_loader), "rmse log": errors["rmse log"]/len(test_loader), "abs rel": errors["abs rel"]/len(test_loader), "sq rel": errors["sq rel"]/len(test_loader)})

if __name__ == '__main__':
    main()