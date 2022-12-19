# --------------------------------------------------------
# Phase Two: Drone Navigation
# Build 2.0.0
# Written by Luc den Ridder
# --------------------------------------------------------

""" Load Libraries, Functions, Classes """
import numpy as np
import wandb
import torch
import torch.nn as nn

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure

from models import SwinTransformer
from models_extra.unet import U_Net


def compute_reprojection_loss(pred, target, device, with_lpips=False):
    """Computes reprojection loss between a batch of predicted and target images
    """
    l1 = torch.nn.L1Loss()
    l1_loss = l1(pred, target)

    if with_lpips == True:
        lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
        lpips_loss = lpips(pred, target)
        lpips.reset()
        reprojection_loss = 0.75*l1_loss+0.25*lpips_loss
    else:
        reprojection_loss = l1_loss
        lpips_loss = 0

    return reprojection_loss, lpips_loss, l1_loss

def data_graft(left_images, right_images):
    rand_w = np.random.randint(0, 4) / 5
    b, c, h, w = left_images.shape
    if int(rand_w * h) == 0:
        return left_images, right_images

    for i in range(b):
        n = list(range(b))
        n.remove(i)
        j = np.random.choice(n)
        graft_h = int(rand_w * h)
        is_flip= np.random.choice([0,1]) == 1
        left_images[i,:,graft_h:] = left_images.clone()[j,:,graft_h:]
        right_images[i,:,graft_h:] = right_images.clone()[j,:,graft_h:]
        if is_flip:
            left_images_tmp = left_images.clone()
            right_images_tmp = right_images.clone()
            left_images[i, :, :-graft_h] = left_images_tmp[i, :, graft_h:]
            left_images[i, :, -graft_h:] = left_images_tmp[i, :, :graft_h]
            right_images[i, :, :-graft_h] = right_images_tmp[i, :, graft_h:]
            right_images[i, :, -graft_h:] = right_images_tmp[i, :, :graft_h]

    return left_images, right_images

def loss_function(left_images, right_images, outputs, cfg, warp_identity,device):
    """Computes the losses during training """
    loss_left = 0
    loss_right = 0

    if warp_identity == 0 and cfg.settings.output_image == 0:
        loss_right, lpips_loss, l1_loss = compute_reprojection_loss(outputs, right_images, device)
    elif warp_identity == 1 and cfg.settings.output_image == 0:
        loss_left, lpips_loss, l1_loss = compute_reprojection_loss(outputs, left_images, device)
    loss = loss_right + loss_left
    
    return loss, lpips_loss, l1_loss, loss_right, loss_left

def loss_function_eval(left_images, right_images, outputs, cfg, cfgdir, device, warp_identity, metrics, metrics_score, losses):
    """" Computes the losses during validation and testing"""
    loss_left = 0
    loss_right = 0
    if warp_identity == 0 and cfg.settings.output_image == 0:
        loss_right, lpips_loss, l1_loss = compute_reprojection_loss(outputs, right_images, device)
        metrics_score = metrics_compute(right_images, outputs, metrics, metrics_score, cfgdir)

    elif warp_identity == 1 and cfg.settings.output_image == 0:
        loss_left, lpips_loss, l1_loss = compute_reprojection_loss(outputs, left_images, device)
        metrics_score = metrics_compute(left_images, outputs, metrics, metrics_score, cfgdir)

    losses["average"] += loss_left + loss_right
    losses["left"] += loss_left
    losses["right"] += loss_right
    losses["LPIPS"] += lpips_loss
    losses["L1"] += l1_loss

    return metrics_score, losses

def mean_std_printer(train_loader):
    """ Prints the mean and standard deviation of the dataset"""
    for left_images,right_images in train_loader:
        mean, std = left_images.mean([0,2,3]), right_images.std([0,2,3])
        print(mean)
        print(std)

def metrics_compute(images, outputs, metrics, metrics_score, cfgdir):
    """ Computes the performance of the model for multiple metrics """
    images = images[:,:,cfgdir.files.image_height_pad:-cfgdir.files.image_height_pad,cfgdir.files.image_width_pad:-cfgdir.files.image_width_pad]
    outputs = outputs[:,:,cfgdir.files.image_height_pad:-cfgdir.files.image_height_pad,cfgdir.files.image_width_pad:-cfgdir.files.image_width_pad]

    metrics["FID"].update(images.type(dtype=torch.uint8), real=True)
    metrics["FID"].update(outputs.type(dtype=torch.uint8), real=False)
    metrics_score["FID"] += metrics["FID"].compute()
    metrics_score["SSIM"] += metrics["SSIM"](outputs,images)                    
    metrics_score["PSNR"] += metrics["PSNR"](outputs,images)                    
    metrics_score["MSSIM"] += metrics["MSSIM"](outputs,images)
    metrics_score["LPIPS"] += metrics["LPIPS"](outputs,images)

    metrics["FID"].reset()
    metrics["SSIM"].reset()
    metrics["PSNR"].reset()
    metrics["MSSIM"].reset()
    metrics["LPIPS"].reset()
    return metrics_score

def metrics_loader(device):
    """ Loads the metrics used to measure performance of the model """
    fid = FrechetInceptionDistance(feature=64).to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    mssim = MultiScaleStructuralSimilarityIndexMeasure(gaussian_kernel = False, kernel_size = 3).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    metrics = {'FID': fid, "SSIM": ssim, "PSNR": psnr, "MSSIM": mssim, "LPIPS": lpips}
    return metrics

def model_loader(cfg, cfgdir, device):
    """ Loads the model """
    if cfg.settings.models[cfg.settings.model_option] == "UNet":
        model = U_Net()
    elif cfg.settings.models[cfg.settings.model_option] == "SWIN":
            model = SwinTransformer(
            img_size = [cfgdir.files.image_height+2*cfgdir.files.image_height_pad, cfgdir.files.image_width+2*cfgdir.files.image_width_pad],
            batch_size = cfgdir.hyperparams.batch_size,
            patch_size = cfgdir.hyperparams.patch_size,
            window_size = cfgdir.hyperparams.window_size,
            output_image = cfg.settings.output_image)

    model= nn.DataParallel(model)
    model = model.to(device)
    return model

def output_function(model, left_images, right_images, warp_identity):
    """ Predicts the outputs from a model to an input """
    if warp_identity == 0:                
        outputs = model(left_images,warp_identity)
    elif warp_identity == 1:
        outputs = model(right_images,warp_identity)
    return outputs


def rendering(left_images,right_images,outputs,warp_identity,epoch,cfg,cfgdir,phase):
    """ Renders relevant images """
    if cfg.settings.rendering==True:
        left_image = left_images.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height + 2*cfgdir.files.image_height_pad, cfgdir.files.image_width+ 2*cfgdir.files.image_width_pad).detach().to(torch.device('cpu')).numpy()[0]
        left_image = left_image.transpose(1,2,0)[cfgdir.files.image_height_pad:-cfgdir.files.image_height_pad, cfgdir.files.image_width_pad:-cfgdir.files.image_width_pad,:]*255
        wandb.log({f"{phase}_left-image": wandb.Image(left_image, caption= f"Epoch {epoch} Left Image")})
        
        right_image = right_images.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height + 2*cfgdir.files.image_height_pad, cfgdir.files.image_width+ 2*cfgdir.files.image_width_pad).detach().to(torch.device('cpu')).numpy()[0]
        right_image = right_image.transpose(1,2,0)[cfgdir.files.image_height_pad:-cfgdir.files.image_height_pad, cfgdir.files.image_width_pad:-cfgdir.files.image_width_pad,:]*255
        wandb.log({f"{phase}_right-image": wandb.Image(right_image, caption= f"Epoch {epoch} Right Image")})

        if warp_identity == 1:    
            predicted_image = outputs.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height + 2*cfgdir.files.image_height_pad, cfgdir.files.image_width+ 2*cfgdir.files.image_width_pad).detach().to(torch.device('cpu')).numpy()[0]
            predicted_image = predicted_image.transpose(1,2,0)[cfgdir.files.image_height_pad:-cfgdir.files.image_height_pad, cfgdir.files.image_width_pad:-cfgdir.files.image_width_pad,:]*255
            wandb.log({f"{phase}_left-image-predicted": wandb.Image(predicted_image, caption= f"Epoch {epoch} Left Image Predicted")})
            
            diff_predleft = np.subtract(predicted_image, left_image)
            wandb.log({f"{phase}_diff-pred-left": wandb.Image(diff_predleft, caption= f"Epoch {epoch} Predicted Difference with Left Image")})
            diff_leftright = np.subtract(left_image, right_image)
            wandb.log({f"{phase}_diff-left-right": wandb.Image(diff_leftright, caption= f"Epoch {epoch} Difference Left and Right Image")})

        elif warp_identity == 0:
            predicted_image = outputs.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height + 2*cfgdir.files.image_height_pad, cfgdir.files.image_width+ 2*cfgdir.files.image_width_pad).detach().to(torch.device('cpu')).numpy()[0]
            predicted_image = predicted_image.transpose(1,2,0)[cfgdir.files.image_height_pad:-cfgdir.files.image_height_pad, cfgdir.files.image_width_pad:-cfgdir.files.image_width_pad,:]*255
            wandb.log({f"{phase}_right-image-predicted": wandb.Image(predicted_image, caption= f"Epoch {epoch} Right Image Predicted")})
            
            diff_predright = np.subtract(predicted_image, right_image)     
            wandb.log({f"{phase}_diff-pred-right": wandb.Image(diff_predright, caption= f"Epoch {epoch} Predicted Difference with Right Image")})
            diff_leftright = np.subtract(left_image, right_image)
            wandb.log({f"{phase}_diff-left-right": wandb.Image(diff_leftright, caption= f"Epoch {epoch} Difference Left and Right Image")})

def save_model(model,run,epoch,cfg,cfgdir):
    """ Saves the model on Weights&Biases """
    artifact = wandb.Artifact(f'{cfg.settings.models[cfg.settings.model_option]}{cfgdir.files.dataloader}_lr-{cfgdir.hyperparams.base_lr}_bs-{cfgdir.hyperparams.batch_size}_epoch-{epoch}', type='model')
    torch.save(model.state_dict(), f'tmp_model.pth')
    artifact.add_file(f'tmp_model.pth')
    run.log_artifact(artifact)

def wandb_loader(cfg, cfgdir,prefix=""):
    """ Loads Weights&Biases """
    run = wandb.init(
                project="Autoencoder (2.0.0)",
                name=f"{prefix}{cfg.settings.models[cfg.settings.model_option]} {cfgdir.files.dataloader} dec:{cfg.settings.output_image} lr={cfgdir.hyperparams.base_lr} bs={cfgdir.hyperparams.batch_size}",
                notes=f"naming convention introduction",
                tags=["2.0.0"],
                config = dict(
                    NameDataset = cfgdir.files.dataloader,
                    Model = cfg.settings.models[cfg.settings.model_option],
                    OutputImage = cfg.settings.output_image,
                    Hyperparameter_LearningRate = cfgdir.hyperparams.base_lr,
                    Hyperparameter_BatchSize = cfgdir.hyperparams.batch_size,
                    Hyperparameter_MeanDataset = cfgdir.hyperparams.mean))
    return run

def wandb_training_logger(loss, lpips_loss, l1_loss, loss_right, loss_left, epoch, i, cfg):
    """ Logs results in training """
    if cfg.settings.wandb==True and i % 10 == 0:
        wandb.log({"Epoch": epoch, "Train Loss": loss, "Train LPIPS Loss": lpips_loss, "Train Loss L1": l1_loss, "Train Loss Right": loss_right, "Train Loss Left": loss_left})