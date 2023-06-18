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
import cv2

# load functions and classses
from dataset import CityScapes, Kitti, Kitti_Tulsiani, Simulation, KittiDepth
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from utils import *

# select gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    test_dataset = dataset['dataloader'](cfgdir, cfgdir.files.test_data)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfgdir.hyperparams.batch_size, shuffle=False)

    # mean_std_printer(train_loader)

    # load model, loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    artifact = run.use_artifact(f'ldenridder/Autoencoder (1.1.4)/SWINKITTI_lr-0.0005_bs-8_epoch-test:v18', type='model')

    artifact.download()
    model = model_loader(cfg, cfgdir, device)
    checkpoint = torch.load(f'artifacts/SWINKITTI_lr-0.0005_bs-8_epoch-test:v18/tmp_model.pth')
    model.load_state_dict(checkpoint, strict=True)
    
    optimizer = build_optimizer(cfgdir, model)

    summary(model, [(cfgdir.hyperparams.batch_size, 3, cfgdir.files.image_height, cfgdir.files.image_width+2*cfgdir.files.image_pad), (1,)], dtypes=[torch.float, torch.int])

    if cfg.settings.wandb==True:
        wandb.watch(model, optimizer, log="all", log_freq=10000)
        

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
            warp_identity = 0
            outputs, depth_gt, depth_outputs, visual_depth_gt, visual_depth_outputs = output_function_depthv3_eval(model, data, cfgdir, device)


            if i%5 == 0 and i<50 and cfg.settings.rendering==True:    
                rendering_depth(data['left image'],data['right image'],outputs,visual_depth_gt,visual_depth_outputs,warp_identity,cfg,cfgdir,i,"swinunet")
                print("Done")
            error = compute_depth_errors(depth_gt, depth_outputs)
            for item in errors:
                errors[item] += error[item]

        wandb.log({"a1": errors["a1"]/len(test_loader), "a2": errors["a2"]/len(test_loader), "a3": errors["a3"]/len(test_loader), "rmse": errors["rmse"]/len(test_loader), "rmse log": errors["rmse log"]/len(test_loader), "abs rel": errors["abs rel"]/len(test_loader), "sq rel": errors["sq rel"]/len(test_loader)})

if __name__ == '__main__':
    main()