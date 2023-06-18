# --------------------------------------------------------
# Step 01: Autoencoder
# Build 1.1.4
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
Build 1.1.4 (29-09-2022)
New features:
- Code Review Ready

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
from dataset import CityScapes, Kitti, Kitti_Tulsiani, Simulation
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

kitti_tulsiani = {
    'csname': "kitti_tulsiani_config",
    'nodename': "KITTITulsianiConfig"}


cityscapes = {
    'csname': "cityscapes_config",
    'nodename': "CityScapesConfig"}

simulation = {
    'csname': "simulation_config",
    'nodename': "SimulationConfig"}

dataset = kitti_tulsiani
cs = ConfigStore.instance()
cs.store(name=dataset['csname'], node=dataset['nodename'])

""" Main Code """
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: dataset['nodename']):
    # initialise weights&biases and check for GPU
    if dataset['csname'] == "kitti_config":
        dataset['dataloader'] = Kitti
        cfgdir = cfg.kitti
    if dataset['csname'] == "kitti_tulsiani_config":
        dataset['dataloader'] = Kitti_Tulsiani
        cfgdir = cfg.kitti_tulsiani
    if dataset['csname'] == "cityscapes_config":
        dataset['dataloader'] = CityScapes
        cfgdir = cfg.cityscapes
    if dataset['csname'] == "simulation_config":
        dataset['dataloader'] = Simulation
        cfgdir = cfg.simulation

    if cfg.settings.wandb==True:
        run = wandb_loader(cfg,cfgdir)
        # artifact = run.use_artifact(f'ldenridder/Autoencoder (1.1.4)/SWINKITTI_lr-{cfgdir.hyperparams.base_lr}_bs-{cfgdir.hyperparams.batch_size}_epoch-test:v3', type='model')
        # artifact.download()        


    # loading in the dataset
    train_dataset = dataset['dataloader'](cfgdir, cfgdir.files.train_data)
    val_dataset = dataset['dataloader'](cfgdir, cfgdir.files.val_data)
    test_dataset = dataset['dataloader'](cfgdir, cfgdir.files.test_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfgdir.hyperparams.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    # mean_std_printer(train_loader)

    # load model, loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


    model = model_loader(cfg, cfgdir, device)
    # checkpoint = torch.load(f'artifacts/SWINSimulation_lr-0.0005_bs-8_epoch-90:v0/tmp_model.pth')
    # model.load_state_dict(checkpoint, strict=False)
    
    optimizer = build_optimizer(cfgdir, model)
    lr_scheduler = build_scheduler(cfgdir.hyperparams, optimizer, len(train_loader))
    metrics = metrics_loader(device)

    summary(model, [(cfgdir.hyperparams.batch_size, 3, cfgdir.files.image_height, cfgdir.files.image_width), (1,)], dtypes=[torch.float, torch.int])

    if cfg.settings.wandb==True:
        wandb.watch(model, optimizer, log="all", log_freq=10000)
        
    """Model Training"""
    for epoch in range(cfgdir.hyperparams.epochs):
        for i, (left_images,right_images) in enumerate(train_loader):
            while np.shape(left_images)[0] != cfgdir.hyperparams.batch_size:
                if np.shape(left_images)[0]*2 < cfgdir.hyperparams.batch_size:
                    left_images = torch.cat((left_images,left_images), dim=0)
                    right_images = torch.cat((right_images,right_images), dim=0)
                else:
                    left_images = torch.cat((left_images,left_images[:cfgdir.hyperparams.batch_size-np.shape(left_images)[0]]), dim=0)
                    right_images = torch.cat((right_images,right_images[:cfgdir.hyperparams.batch_size-np.shape(right_images)[0]]), dim=0)
            left_images = left_images.float().to(device)
            right_images = right_images.float().to(device)            

            """Main Model"""
            # Forward pass
            warp_identity = np.random.choice([0,1])
            outputs = output_function(model, left_images, right_images, warp_identity)
            loss, lpips_loss, l1_loss, loss_right, loss_left = loss_function(left_images, right_images, outputs, cfg, warp_identity, device)

            # Backward and optimize
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * len(train_loader) + i))
            loss.backward()
            optimizer.step()

            wandb_training_logger(loss, lpips_loss, l1_loss, loss_right, loss_left, epoch, i, cfg)

        """W&B Information"""    
        if epoch % 10 == 0:
            rendering(left_images,right_images,outputs,warp_identity,epoch,cfg,cfgdir,"training")
            save_model(model,run,epoch,cfg,cfgdir)
            print (f'Epoch [{epoch+1}/{cfgdir.hyperparams.epochs}], Loss: {loss.item():.4f}')

            with torch.no_grad():
                losses = {"average": 0, "left":0, "right":0, "SSIM": 0, "L1": 0, "LPIPS": 0}
                metrics_score = {"FID": 0, "SSIM": 0, "PSNR": 0, "MSSIM": 0, "LPIPS": 0}

                for i, (left_images, right_images) in enumerate(val_loader):
                    left_images = left_images.float().to(device)
                    right_images = right_images.float().to(device)

                    warp_identity = i % 2
                    outputs = output_function(model, left_images, right_images, warp_identity)
                    metrics_score, losses = loss_function_eval(left_images, right_images, outputs, cfg, cfgdir, device, warp_identity, metrics, metrics_score, losses)
                    
                    wandb.log({"Epoch": epoch})
                    if i == 0:
                        rendering(left_images,right_images,outputs,warp_identity,epoch,cfg,cfgdir,"validation")
                        
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
            
            warp_identity = i % 2
            outputs = output_function(model, left_images, right_images, warp_identity)
            metrics_score, losses = loss_function_eval(left_images, right_images, outputs, cfg, cfgdir, device, warp_identity, metrics, metrics_score, losses)
            
            if i == 0:
                rendering(left_images,right_images,outputs,warp_identity,0,cfg,cfgdir,"testing")
                save_model(model,run,'test',cfg, cfgdir)

        wandb.log({"Validation (+Test) Loss": losses['average']/len(test_loader), "Validation (+Test) Loss LPIPS": losses['LPIPS']/len(test_loader), "Validation (+Test) Loss L1": losses['L1']/len(test_loader),
            "FID Score": metrics_score["FID"]/len(test_loader), "SSIM Score": metrics_score["SSIM"]/len(test_loader), "PSNR Score": metrics_score["PSNR"]/len(test_loader), "MSSIM Score": metrics_score["MSSIM"]/len(test_loader), "LPIPS Score": metrics_score["LPIPS"]/len(test_loader)})

if __name__ == '__main__':
    main()