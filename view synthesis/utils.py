from locale import normalize
import wandb
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
import torch
from models import SwinTransformer, SwinTransformerDepth, SwinTransformer_cfg1, SwinTransformer_cfg2, SwinTransformer_simple, SwinTransformer_wh_skipv2
from models_testing import SwinTransformer_Layer0, SwinTransformer_Layer1, SwinTransformer_Layer2, SwinTransformer_Layer3, SwinTransformer_Wo1Skip, SwinTransformer_Wo2Skip, SwinTransformer_Wo3Skip, SwinTransformer_WoAllSkip, SwinTransformer_1Skip ,SwinTransformer_2Skip, SwinTransformer_3Skip
from models_skip import SwinTransformer_excl1, SwinTransformer_excl2, SwinTransformer_excl4, SwinTransformer_excl8, SwinTransformer_exclbn, SwinTransformer_bn, SwinTransformer_skip8, SwinTransformer_skip4, SwinTransformer_skip2, SwinTransformer_skip1
from models_skip_train import SwinTransformer_bn_train
from models_extra.unet import U_Net, U_NetDepth, U_Net_NoSkip
from models_extra.vnet import VNet
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import seaborn as sns
import cv2
from sgm import Paths, compute_costs, aggregate_costs, select_disparity
import torch.nn.functional as F
from scipy import ndimage
from PIL import Image

def wandb_loader(cfg, cfgdir,prefix=""):
    run = wandb.init(
                project="Autoencoder (1.1.4)",
                name=f"NoSkip-{prefix}{cfg.settings.models[cfg.settings.model_option]} {cfgdir.files.dataloader} dec:{cfg.settings.output_image} lr={cfgdir.hyperparams.base_lr} bs={cfgdir.hyperparams.batch_size}",
                notes=f"naming convention introduction",
                tags=["1.1.4"],
                config = dict(
                    NameDataset = cfgdir.files.dataloader,
                    Model = cfg.settings.models[cfg.settings.model_option],
                    OutputImage = cfg.settings.output_image,
                    Hyperparameter_LearningRate = cfgdir.hyperparams.base_lr,
                    Hyperparameter_BatchSize = cfgdir.hyperparams.batch_size,
                    Hyperparameter_MinLR = cfgdir.hyperparams.min_lr,
                    Hyperparameter_WarmUpEp = cfgdir.hyperparams.warmup_epochs))
    return run

def wandb_training_logger(loss, lpips_loss, l1_loss, loss_right, loss_left, epoch, i, cfg):
    if cfg.settings.wandb==True and i % 10 == 0:
        if cfg.settings.output_image == 0:
            wandb.log({"Epoch": epoch, "Train Loss": loss, "Train LPIPS Loss": lpips_loss, "Train Loss L1": l1_loss, "Train Loss Right": loss_right, "Train Loss Left": loss_left})
        elif cfg.settings.output_image == 1:
            wandb.log({"Epoch": epoch, "Train Multiple Loss": loss, "Train Original Loss": loss_original, "Train Warped Loss": loss_warp})


def mean_std_printer(train_loader):
    for left_images,right_images in train_loader:
        mean, std = left_images.mean([0,2,3]), right_images.std([0,2,3])
        print(mean)
        print(std)

def mean_std_depth_printer(train_loader):
    for i, (left_images,right_images, depth_gt) in enumerate(train_loader):
        mean, std = depth_gt.mean(), depth_gt.std()
        print(mean)
        print(std)

def metrics_loader(device):
    fid = FrechetInceptionDistance(feature=64).to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    mssim = MultiScaleStructuralSimilarityIndexMeasure(gaussian_kernel = False, kernel_size = 3).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True,net_type='vgg').to(device)

    metrics = {'FID': fid, "SSIM": ssim, "PSNR": psnr, "MSSIM": mssim, "LPIPS": lpips}
    return metrics

def model_loader(cfg, cfgdir, device):
    if cfg.settings.models[cfg.settings.model_option] == "UNet":
        model = U_Net_NoSkip()
    elif cfg.settings.models[cfg.settings.model_option] == "VNet":
        model = VNet()
    elif cfg.settings.models[cfg.settings.model_option] == "SWIN":
        model = SwinTransformer_cfg1(
            img_size = cfgdir.swinparams.img_size,
            batch_size = cfgdir.hyperparams.batch_size,
            patch_size = cfgdir.swinparams.patch_size,
            window_size = cfgdir.swinparams.window_size,
            embed_dim = cfgdir.swinparams.embed_dim,
            output_image = cfg.settings.output_image)

    model= nn.DataParallel(model)
    model = model.to(device)
    return model

def model_loader_skip(cfg, cfgdir, device, skip):
    if skip == 1:
            model = SwinTransformer_excl1(
            img_size = cfgdir.swinparams.img_size,
            batch_size = cfgdir.hyperparams.batch_size,
            patch_size = cfgdir.swinparams.patch_size,
            window_size = cfgdir.swinparams.window_size,
            output_image = cfg.settings.output_image)

    if skip == 2:
            model = SwinTransformer_excl2(
            img_size = cfgdir.swinparams.img_size,
            batch_size = cfgdir.hyperparams.batch_size,
            patch_size = cfgdir.swinparams.patch_size,
            window_size = cfgdir.swinparams.window_size,
            output_image = cfg.settings.output_image)

    if skip == 4:
            model = SwinTransformer_excl4(
            img_size = cfgdir.swinparams.img_size,
            batch_size = cfgdir.hyperparams.batch_size,
            patch_size = cfgdir.swinparams.patch_size,
            window_size = cfgdir.swinparams.window_size,
            output_image = cfg.settings.output_image)

    if skip == 8:
            model = SwinTransformer_excl8(
            img_size = cfgdir.swinparams.img_size,
            batch_size = cfgdir.hyperparams.batch_size,
            patch_size = cfgdir.swinparams.patch_size,
            window_size = cfgdir.swinparams.window_size,
            output_image = cfg.settings.output_image)

    if skip == 0:
            model = SwinTransformer_exclbn(
            img_size = cfgdir.swinparams.img_size,
            batch_size = cfgdir.hyperparams.batch_size,
            patch_size = cfgdir.swinparams.patch_size,
            window_size = cfgdir.swinparams.window_size,
            output_image = cfg.settings.output_image)

    if skip == -1:
            model = SwinTransformer_skip1(
            img_size = cfgdir.swinparams.img_size,
            batch_size = cfgdir.hyperparams.batch_size,
            patch_size = cfgdir.swinparams.patch_size,
            window_size = cfgdir.swinparams.window_size,
            output_image = cfg.settings.output_image)

    if skip == -2:
            model = SwinTransformer_skip2(
            img_size = cfgdir.swinparams.img_size,
            batch_size = cfgdir.hyperparams.batch_size,
            patch_size = cfgdir.swinparams.patch_size,
            window_size = cfgdir.swinparams.window_size,
            output_image = cfg.settings.output_image)

    if skip == -4:
            model = SwinTransformer_skip4(
            img_size = cfgdir.swinparams.img_size,
            batch_size = cfgdir.hyperparams.batch_size,
            patch_size = cfgdir.swinparams.patch_size,
            window_size = cfgdir.swinparams.window_size,
            output_image = cfg.settings.output_image)

    if skip == -8:
            model = SwinTransformer_skip8(
            img_size = cfgdir.swinparams.img_size,
            batch_size = cfgdir.hyperparams.batch_size,
            patch_size = cfgdir.swinparams.patch_size,
            window_size = cfgdir.swinparams.window_size,
            output_image = cfg.settings.output_image)

    if skip == 100:
            model = SwinTransformer_bn(
            img_size = cfgdir.swinparams.img_size,
            batch_size = cfgdir.hyperparams.batch_size,
            patch_size = cfgdir.swinparams.patch_size,
            window_size = cfgdir.swinparams.window_size,
            output_image = cfg.settings.output_image)

    model= nn.DataParallel(model)
    model = model.to(device)
    return model

def model_loader_depth(cfg,cfgdir, device):
    if cfg.settings.models[cfg.settings.model_option] == "UNet":
        model = U_NetDepth()
    elif cfg.settings.models[cfg.settings.model_option] == "SWIN":
        model = SwinTransformerDepth(
            img_size = cfgdir.swinparams.img_size,
            patch_size = cfgdir.swinparams.patch_size,
            window_size = cfgdir.swinparams.window_size)
    
    model= nn.DataParallel(model)
    model = model.to(device)
    return model

def model_loader_test(cfg, cfgdir, device,n_layer):
    if cfg.settings.models[cfg.settings.model_option] == "UNet":
        model = U_Net()
    elif cfg.settings.models[cfg.settings.model_option] == "VNet":
        model = VNet()
    elif cfg.settings.models[cfg.settings.model_option] == "SWIN":
        if n_layer == 0:
            model = SwinTransformer_Layer0(
                img_size = cfgdir.swinparams.img_size,
                batch_size = cfgdir.hyperparams.batch_size,
                patch_size = cfgdir.swinparams.patch_size,
                window_size = cfgdir.swinparams.window_size,
                output_image = cfg.settings.output_image)


        if n_layer == 1:
            model = SwinTransformer_Layer1(
                img_size = cfgdir.swinparams.img_size,
                batch_size = cfgdir.hyperparams.batch_size,
                patch_size = cfgdir.swinparams.patch_size,
                window_size = cfgdir.swinparams.window_size,
                output_image = cfg.settings.output_image)
        
        if n_layer == 2:
            model = SwinTransformer_Layer2(
                img_size = cfgdir.swinparams.img_size,
                batch_size = cfgdir.hyperparams.batch_size,
                patch_size = cfgdir.swinparams.patch_size,
                window_size = cfgdir.swinparams.window_size,
                output_image = cfg.settings.output_image)

        if n_layer == 3:
            model = SwinTransformer_Layer3(
                img_size = cfgdir.swinparams.img_size,
                batch_size = cfgdir.hyperparams.batch_size,
                patch_size = cfgdir.swinparams.patch_size,
                window_size = cfgdir.swinparams.window_size,
                output_image = cfg.settings.output_image)

        if n_layer == 5:
            model = SwinTransformer_Wo1Skip(
                img_size = cfgdir.swinparams.img_size,
                batch_size = cfgdir.hyperparams.batch_size,
                patch_size = cfgdir.swinparams.patch_size,
                window_size = cfgdir.swinparams.window_size,
                output_image = cfg.settings.output_image)

        if n_layer == 6:
            model = SwinTransformer_Wo2Skip(
                img_size = cfgdir.swinparams.img_size,
                batch_size = cfgdir.hyperparams.batch_size,
                patch_size = cfgdir.swinparams.patch_size,
                window_size = cfgdir.swinparams.window_size,
                output_image = cfg.settings.output_image)

        if n_layer == 7:
            model = SwinTransformer_Wo3Skip(
                img_size = cfgdir.swinparams.img_size,
                batch_size = cfgdir.hyperparams.batch_size,
                patch_size = cfgdir.swinparams.patch_size,
                window_size = cfgdir.swinparams.window_size,
                output_image = cfg.settings.output_image)
        
        if n_layer == 8:
            model = SwinTransformer_WoAllSkip(
                img_size = cfgdir.swinparams.img_size,
                batch_size = cfgdir.hyperparams.batch_size,
                patch_size = cfgdir.swinparams.patch_size,
                window_size = cfgdir.swinparams.window_size,
                output_image = cfg.settings.output_image)

        if n_layer == 9:
            model = SwinTransformer_1Skip(
                img_size = cfgdir.swinparams.img_size,
                batch_size = cfgdir.hyperparams.batch_size,
                patch_size = cfgdir.swinparams.patch_size,
                window_size = cfgdir.swinparams.window_size,
                output_image = cfg.settings.output_image)

        if n_layer == 10:
            model = SwinTransformer_2Skip(
                img_size = cfgdir.swinparams.img_size,
                batch_size = cfgdir.hyperparams.batch_size,
                patch_size = cfgdir.swinparams.patch_size,
                window_size = cfgdir.swinparams.window_size,
                output_image = cfg.settings.output_image)
        
        if n_layer == 11:
            model = SwinTransformer_3Skip(
                img_size = cfgdir.swinparams.img_size,
                batch_size = cfgdir.hyperparams.batch_size,
                patch_size = cfgdir.swinparams.patch_size,
                window_size = cfgdir.swinparams.window_size,
                output_image = cfg.settings.output_image)
        
    model= nn.DataParallel(model)
    model = model.to(device)
    return model

def output_function(model, left_images, right_images, warp_identity):
    if warp_identity == 0:                
        outputs = model(left_images,warp_identity)
    elif warp_identity == 1:
        outputs = model(right_images,warp_identity)
    return outputs

def output_function_depth(model, left_images, right_images, left_depth_gt, right_depth_gt, warp_identity, cfgdir, device):
    if warp_identity == 0:                
        outputs = model(left_images)
        depth_gt = right_depth_gt
    elif warp_identity == 1:
        outputs = model(right_images)
        depth_gt = left_depth_gt

    
    # vmax_left = np.percentile(depth_gt.view(8, 375, 1242).detach().to(torch.device('cpu')).numpy()[0], 99)
    # plt.figure()
    # plt.imshow(depth_gt.view(8, 375, 1242).detach().to(torch.device('cpu')).numpy()[0], cmap='magma', vmax=vmax_left)
    # plt.axis('off')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig('depth_gt.png', bbox_inches='tight',pad_inches= 0)
    # plt.close()

    # depth_gt = F.interpolate(depth_gt, (375,1280), mode="bilinear", align_corners=True)
    # print(depth_gt.shape)
    # vmax_left = np.percentile(depth_gt.view(8, 375, 1280).detach().to(torch.device('cpu')).numpy()[0], 99)
    # plt.figure()
    # plt.imshow(depth_gt.view(8, 375, 1280).detach().to(torch.device('cpu')).numpy()[0], cmap='magma', vmax=vmax_left)
    # plt.axis('off')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig('depth_gt_s.png', bbox_inches='tight',pad_inches= 0)
    # plt.close()

    outputs = outputs[:,:,:,2*cfgdir.files.image_pad:-2*cfgdir.files.image_pad]
    outputs = F.interpolate(outputs, (depth_gt.shape[2], depth_gt.shape[3]), mode="bilinear", align_corners=False)

    visual_outputs = outputs.detach().clone()
    outputs = torch.masked_select(outputs, depth_gt.ne(torch.zeros(depth_gt.shape).to(device)))
    visual_depth_gt = depth_gt.detach().clone()
    depth_gt = torch.masked_select(depth_gt, depth_gt.ne(torch.zeros(depth_gt.shape).to(device)))
    return outputs, depth_gt, visual_outputs, visual_depth_gt

def output_function_depth_eval(model, data, warp_identity, cfgdir, device):
    if warp_identity == 0:                
        outputs = model(data['left image'])
        depth_gt = data['right depth gt']
        valid_mask = data['right valid mask']
        stereo_T = data["stereo_T-l"]
    elif warp_identity == 1:
        outputs = model(data['right image'])
        depth_gt = data['left depth gt']
        valid_mask = data['left valid mask']
        stereo_T = data["stereo_T-r"]

    
    # vmax_left = np.percentile(depth_gt.view(8, 375, 1242).detach().to(torch.device('cpu')).numpy()[0], 99)
    # plt.figure()
    # plt.imshow(depth_gt.view(8, 375, 1242).detach().to(torch.device('cpu')).numpy()[0], cmap='magma', vmax=vmax_left)
    # plt.axis('off')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig('depth_gt.png', bbox_inches='tight',pad_inches= 0)
    # plt.close()

    # depth_gt = F.interpolate(depth_gt, (375,1280), mode="bilinear", align_corners=True)
    # print(depth_gt.shape)
    # vmax_left = np.percentile(depth_gt.view(8, 375, 1280).detach().to(torch.device('cpu')).numpy()[0], 99)
    # plt.figure()
    # plt.imshow(depth_gt.view(8, 375, 1280).detach().to(torch.device('cpu')).numpy()[0], cmap='magma', vmax=vmax_left)
    # plt.axis('off')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig('depth_gt_s.png', bbox_inches='tight',pad_inches= 0)
    # plt.close()

    outputs = outputs[:,:,:,2*cfgdir.files.image_pad:-2*cfgdir.files.image_pad]
    outputs =generate_warp_image

    outputs = F.interpolate(outputs, (depth_gt.shape[2], depth_gt.shape[3]), mode="bilinear", align_corners=False)

    outputs[valid_mask == False] == 0
    depth_gt[valid_mask == False] == 0

    visual_outputs = outputs.detach().clone()
    outputs = torch.masked_select(outputs, depth_gt.ne(torch.zeros(depth_gt.shape).to(device)))
    visual_depth_gt = depth_gt.detach().clone()
    depth_gt = torch.masked_select(depth_gt, depth_gt.ne(torch.zeros(depth_gt.shape).to(device)))
    return outputs, depth_gt, visual_outputs, visual_depth_gt

def output_function_depthv2_eval(model, data, warp_identity, cfgdir, device):
    if warp_identity == 0:                
        outputs = model(data['right image'])
        depth_gt = data['right depth gt']
        valid_mask = data['right valid mask']
        stereo_T = data["stereo_T-r"]
        image = data['left image hq']
    elif warp_identity == 1:
        outputs = model(data['left image'])
        depth_gt = data['left depth gt']
        valid_mask = data['left valid mask']
        stereo_T = data["stereo_T-l"]
        image = data['right image hq']
    
    outputs = outputs[:,:,:,2*cfgdir.files.image_pad:-2*cfgdir.files.image_pad]
    outputs[outputs < 0] == 0
    depth_outputs = outputs.detach().clone()

    depth_outputs = F.interpolate(depth_outputs, (depth_gt.shape[2], depth_gt.shape[3]), mode="bilinear", align_corners=False)

    visual_depth_outputs = depth_outputs.detach().clone()
    visual_depth_gt = depth_gt.detach().clone()
    visual_depth_outputs[valid_mask == False] == 0
    visual_depth_gt[valid_mask == False] == 0

    depth_outputs  = torch.masked_select(depth_outputs, valid_mask)
    depth_gt = torch.masked_select(depth_gt,valid_mask)


    outputs = generate_warp_image(image,data["K"],stereo_T,outputs)
  
    return outputs, depth_gt, depth_outputs, visual_depth_gt, visual_depth_outputs

def output_function_depthv3_eval(model, data, cfgdir, device):

             
    outputs = model(data['left image'],0)
    depth_gt = data['left depth gt']
    valid_mask = data['left valid mask']
    stereo_T = data["stereo_T-r"]
    image = data['left image hq']
    outputs = outputs[:,:,:,cfgdir.files.image_pad:-cfgdir.files.image_pad]
    left_images = data['left image'].detach().to(torch.device('cpu'))*255
    left_images = left_images.numpy().astype(dtype=np.uint8)[:,:,:,cfgdir.files.image_pad:-cfgdir.files.image_pad]
        
    # outputs_numpy = data['right image'].detach().to(torch.device('cpu'))*255
    # outputs_numpy = outputs_numpy.numpy().astype(dtype=np.uint8)[:,:,:,cfgdir.files.image_pad:-cfgdir.files.image_pad]
    outputs_numpy = outputs.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width).detach().to(torch.device('cpu'))*255
    outputs_numpy = outputs_numpy.numpy().astype(dtype=np.uint8)

    depth_maps = []
    for i in range(len(left_images)):

        right_gray = cv2.cvtColor(outputs_numpy[i].transpose(1,2,0), cv2.COLOR_RGB2GRAY)
        left_gray = cv2.cvtColor(left_images[i].transpose(1,2,0), cv2.COLOR_RGB2GRAY)

        focal_length = 721*(320-32)/1242
        baseline = 0.54

        depth_map = compute_depth_map(left_gray, right_gray, focal_length, baseline)
        depth_maps.append(depth_map)

    depth_outputs = F.interpolate(torch.from_numpy(np.expand_dims(np.array(depth_maps), axis=1)), (depth_gt.shape[2], depth_gt.shape[3]), mode="bilinear", align_corners=False).to(device)
    visual_depth_outputs = depth_outputs.detach().clone()
    visual_depth_gt = depth_gt.detach().clone()
    visual_depth_outputs[valid_mask == False] == 0
    visual_depth_gt[valid_mask == False] == 0

    depth_outputs  = torch.masked_select(depth_outputs, valid_mask)
    depth_gt = torch.masked_select(depth_gt,valid_mask)

    # outputs = generate_warp_image(image,data["K"],stereo_T,outputs)
  
    return outputs, depth_gt, depth_outputs, visual_depth_gt, visual_depth_outputs

def compute_depth_map(left_gray, right_gray, focal_length, baseline):
    # Set up StereoSGBM matcher with appropriate parameters
    window_size = 3
    min_disp = 0
    num_disp = 32

    left_stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=1,
        speckleWindowSize=1,
        speckleRange=1,
        preFilterCap=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_stereo = cv2.StereoSGBM_create(
        minDisparity=-num_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=1,
        speckleWindowSize=1,
        speckleRange=1,
        preFilterCap=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )


    # Compute the left-to-right and right-to-left disparity maps
    disparity_left = left_stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity_right = abs(right_stereo.compute(right_gray, left_gray).astype(np.float32) / 16.0)
    
    # Perform left-right consistency check
    disparity = np.zeros_like(disparity_left)
    disparity[:,:num_disp] = disparity_right[:,:num_disp]
    disparity[:,-num_disp:] = disparity_left[:,-num_disp:]
    disparity[:,num_disp:-num_disp] = (disparity_left[:,num_disp:-num_disp]+disparity_right[:,num_disp:-num_disp])/2

    # disparity[:,:int(disparity.shape[1]/2)] = disparity_right[:,:int(disparity.shape[1]/2)]
    # disparity[:,int(disparity.shape[1]/2):] = disparity_left[:,int(disparity.shape[1]/2):]

    # for y in range(height):
    #     for x in range(width):
    #         disp_value = disparity_left[y, x]
    #         r_x = int(x - disp_value)
            
    #         # Add bounds check for r_x
    #         if r_x >= 0 and r_x < width and abs(disparity_right[y, r_x] - disp_value) <= left_right_diff:
    #             disparity_left_consistent[y, x] = disp_value
    depth_map = focal_length * baseline / (disparity + 1e-10)

    # depth_map[depth_map > 20] = 20
    # depth_map = depth_map/np.max(depth_map)*20
    depth_map[depth_map < 0] = np.min(depth_map)

    return depth_map

def compute_reprojection_loss(pred, target, device, with_lpips=False):
    """Computes reprojection loss between a batch of predicted and target images
    """
    l1 = torch.nn.L1Loss()
    lpips_loss = 0
    if with_lpips == True:
        lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
        lpips_loss = lpips(pred, target)
        lpips.reset()
    # pred = (pred-torch.min(pred))/(torch.max(pred)-torch.min(pred))
    # target = (target-torch.min(target))/(torch.max(target)-torch.min(target))
    #ssim = StructuralSimilarityIndexMeasure(gaussian_kernel = False, kernel_size = 3).to(device)
    l1_loss = l1(pred, target)
    
    ssim_loss = 0 #1-ssim(pred, target)
    reprojection_loss = l1_loss # 0.75*l1_loss+0.25*lpips_loss#0.15 * ssim_loss + 0.85 * l1_loss
    return reprojection_loss, lpips_loss, l1_loss


def loss_function(left_images, right_images, outputs, cfg, warp_identity,device):
    # if warp_identity == 0 and cfg.settings.output_image == 0:
    #     loss = criterion(outputs, right_images)
    # elif warp_identity == 1 and cfg.settings.output_image == 0:
    #     loss = criterion(outputs, left_images)
    # elif warp_identity == 1 and cfg.settings.output_image == 1:
    #     both_images = torch.stack((left_images,right_images), dim=1)
    #     loss = criterion(outputs, both_images)
    #     loss_original = criterion(outputs.permute(1,0,2,3,4)[0], left_images)
    #     loss_warp = criterion(outputs.permute(1,0,2,3,4)[1], right_images)
    # elif warp_identity == 0 and cfg.settings.output_image == 1:
    #     both_images = torch.stack((right_images,left_images), dim=1)
    #     loss = criterion(outputs, both_images)
    #     loss_original = criterion(outputs.permute(1,0,2,3,4)[0], right_images)
    #     loss_warp = criterion(outputs.permute(1,0,2,3,4)[1], left_images)
    # return loss
    loss_left = 0
    loss_right = 0

    if warp_identity == 0 and cfg.settings.output_image == 0:
        loss_right, lpips_loss, l1_loss = compute_reprojection_loss(outputs, right_images, device)
    elif warp_identity == 1 and cfg.settings.output_image == 0:
        loss_left, lpips_loss, l1_loss = compute_reprojection_loss(outputs, left_images, device)
    loss = loss_right + loss_left
    
    return loss, lpips_loss, l1_loss, loss_right, loss_left


def loss_function_depth(data, outputs, cfg, warp_identity,device):
    loss_left = 0
    loss_right = 0

    if warp_identity == 0 and cfg.settings.output_image == 0:
        loss_right, lpips_loss, l1_loss = compute_reprojection_loss(outputs, data["right image hq"], device)
    elif warp_identity == 1 and cfg.settings.output_image == 0:
        loss_left, lpips_loss, l1_loss = compute_reprojection_loss(outputs, data["left image hq"], device)
    loss = loss_right + loss_left
    
    return loss, lpips_loss, l1_loss, loss_right, loss_left


def metrics_compute(images, outputs, metrics, metrics_score, cfgdir):
    
    # images = images[:,:,:,cfgdir.files.image_pad+cfgdir.files.image_extended:-cfgdir.files.image_pad-cfgdir.files.image_extended]
    # outputs = outputs[:,:,:,cfgdir.files.image_pad+cfgdir.files.image_extended:-cfgdir.files.image_pad-cfgdir.files.image_extended]
    # metrics["FID"].update(images.type(dtype=torch.uint8), real=True)
    # metrics["FID"].update(outputs.type(dtype=torch.uint8), real=False)
    # metrics_score["FID"] += metrics["FID"].compute()
    metrics_score["SSIM"] += metrics["SSIM"](outputs,images)                    
    metrics_score["PSNR"] += metrics["PSNR"](outputs,images)                    
    metrics_score["MSSIM"] += metrics["MSSIM"](outputs,images)
    metrics_score["LPIPS"] += metrics["LPIPS"](outputs,images)

    # metrics["FID"].reset()
    metrics["SSIM"].reset()
    metrics["PSNR"].reset()
    metrics["MSSIM"].reset()
    metrics["LPIPS"].reset()
    return metrics_score

def lpips_prepare(images):
    lpips_images = images.detach().clone()
    lpips_images = (lpips_images-torch.min(lpips_images))/(torch.max(lpips_images)-torch.min(lpips_images))
    return lpips_images
    

def loss_function_eval(left_images, right_images, outputs, cfg, cfgdir, device, warp_identity, metrics, metrics_score, losses):
    
    loss_left = 0
    loss_right = 0

    if warp_identity == 0 and cfg.settings.output_image == 0:
        loss_right, lpips_loss, l1_loss = compute_reprojection_loss(outputs, right_images, device)
        metrics_score = metrics_compute(right_images, outputs, metrics, metrics_score, cfgdir)

    elif warp_identity == 1 and cfg.settings.output_image == 0:
        loss_left, lpips_loss, l1_loss = compute_reprojection_loss(outputs, left_images, device)
        metrics_score = metrics_compute(left_images, outputs, metrics, metrics_score, cfgdir)
   
    # left_images_resized = []
    # right_images_resized = []
    # outputs_resized = []

    # for i in range(cfgdir.hyperparams.batch_size):
    #     left_image_resized = np.clip(ndimage.zoom(left_images.cpu()[i], (1,0.5,0.5), order=3),0,1)
    #     right_image_resized = np.clip(ndimage.zoom(right_images.cpu()[i], (1,0.5,0.5), order=3),0,1)
    #     output_resized = np.clip(ndimage.zoom(outputs.cpu()[i], (1,0.5,0.5), order=3),0,1)

    #     left_images_resized.append(left_image_resized)
    #     right_images_resized.append(right_image_resized)
    #     outputs_resized.append(output_resized)

    # left_images_resized = torch.Tensor(np.array(left_images_resized)).to(device)
    # right_images_resized = torch.Tensor(np.array(right_images_resized)).to(device)
    # outputs_resized = torch.Tensor(np.array(outputs_resized)).to(device)

    # if warp_identity == 0 and cfg.settings.output_image == 0:
    #     loss_right, lpips_loss, l1_loss = compute_reprojection_loss(outputs_resized, right_images_resized, device)
    #     metrics_score = metrics_compute(right_images_resized, outputs_resized, metrics, metrics_score, cfgdir)

    # elif warp_identity == 1 and cfg.settings.output_image == 0:
    #     loss_left, lpips_loss, l1_loss = compute_reprojection_loss(outputs_resized, left_images_resized, device)
    #     metrics_score = metrics_compute(left_images_resized, outputs_resized, metrics, metrics_score, cfgdir)

    # elif warp_identity == 1 and cfg.settings.output_image == 1:
    #     both_images = torch.stack((left_images,right_images), dim=1)
    #     loss = criterion(outputs, both_images)
    #     loss_original = criterion(outputs.permute(1,0,2,3,4)[0], left_images)
    #     loss_warp = criterion(outputs.permute(1,0,2,3,4)[1], right_images)
    # elif warp_identity == 0 and cfg.settings.output_image == 1:
    #     both_images = torch.stack((right_images,left_images), dim=1)
    #     loss = criterion(outputs, both_images)
    #     loss_original = criterion(outputs.permute(1,0,2,3,4)[0], right_images)
    #     loss_warp = criterion(outputs.permute(1,0,2,3,4)[1], left_images)
    
    losses["average"] += loss_left + loss_right
    losses["left"] += loss_left
    losses["right"] += loss_right
    losses["LPIPS"] += lpips_loss
    losses["L1"] += l1_loss

    # if cfg.settings.output_image == 1:
    #     total_loss_original += loss_original
    #     total_loss_warp += loss_warp

    return metrics_score, losses

def rendering(left_images,right_images,outputs,warp_identity,epoch,cfg,cfgdir,phase):

    if cfg.settings.rendering==True:
        left_image = left_images.detach().to(torch.device('cpu')).numpy()[0]
        #left_image = left_image.transpose(1,2,0)[:,cfgdir.files.image_pad+cfgdir.files.image_extended:-cfgdir.files.image_pad-cfgdir.files.image_extended,:]*255
        left_image = left_image.transpose(1,2,0)[:,:,:]*255
        wandb.log({f"{phase}_left-image": wandb.Image(left_image, caption= f"Epoch {epoch} Left Image")})
        
        right_image = right_images.detach().to(torch.device('cpu')).numpy()[0]
        #right_image = right_image.transpose(1,2,0)[:,cfgdir.files.image_pad+cfgdir.files.image_extended:-cfgdir.files.image_pad-cfgdir.files.image_extended,:]*255        
        right_image = right_image.transpose(1,2,0)[:,:,:]*255
        wandb.log({f"{phase}_right-image": wandb.Image(right_image, caption= f"Epoch {epoch} Right Image")})

        if warp_identity == 1:    
            predicted_image = outputs.detach().to(torch.device('cpu')).numpy()[0]
            #predicted_image = predicted_image.transpose(1,2,0)[:,cfgdir.files.image_pad+cfgdir.files.image_extended:-cfgdir.files.image_pad-cfgdir.files.image_extended,:]*255      
            predicted_image = predicted_image.transpose(1,2,0)[:,:,:]*255
            wandb.log({f"{phase}_left-image-predicted": wandb.Image(predicted_image, caption= f"Epoch {epoch} Left Image Predicted")})
            
            diff_predleft = np.subtract(predicted_image, left_image)
            wandb.log({f"{phase}_diff-pred-left": wandb.Image(diff_predleft, caption= f"Epoch {epoch} Predicted Difference with Left Image")})
            diff_leftright = np.subtract(left_image, right_image)
            wandb.log({f"{phase}_diff-left-right": wandb.Image(diff_leftright, caption= f"Epoch {epoch} Difference Left and Right Image")})

        elif warp_identity == 0:
            predicted_image = outputs.detach().to(torch.device('cpu')).numpy()[0]
            #predicted_image = predicted_image.transpose(1,2,0)[:,cfgdir.files.image_pad+cfgdir.files.image_extended:-cfgdir.files.image_pad-cfgdir.files.image_extended,:]*255      
            predicted_image = predicted_image.transpose(1,2,0)[:,:,:]*255
            wandb.log({f"{phase}_right-image-predicted": wandb.Image(predicted_image, caption= f"Epoch {epoch} Right Image Predicted")})
            
            diff_predright = np.subtract(predicted_image, right_image)     
            wandb.log({f"{phase}_diff-pred-right": wandb.Image(diff_predright, caption= f"Epoch {epoch} Predicted Difference with Right Image")})
            diff_leftright = np.subtract(left_image, right_image)
            wandb.log({f"{phase}_diff-left-right": wandb.Image(diff_leftright, caption= f"Epoch {epoch} Difference Left and Right Image")})


        elif cfg.settings.output_image == 2:
            predicted_image = outputs.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width+ 2*cfgdir.files.image_pad).detach().to(torch.device('cpu')).numpy()[0]
            predicted_image_left = predicted_image[0]
            predicted_image_right = predicted_image[1]
            predicted_image_left = predicted_image_left.transpose(1,2,0)[:, cfgdir.files.image_pad:-cfgdir.files.image_pad,:]*255      
            predicted_image_right = predicted_image_right.transpose(1,2,0)[:, cfgdir.files.image_pad:-cfgdir.files.image_pad,:]*255      
            wandb.log({f"{phase}_left-image-predicted": wandb.Image(predicted_image_left, caption= f"Epoch {epoch} Left Image Predicted")})
            wandb.log({f"{phase}_right-image-predicted": wandb.Image(predicted_image_right, caption= f"Epoch {epoch} Right Image Predicted")})
            
        
            diff_predright = np.subtract(predicted_image_right, right_image)
            diff_predright_scale = ((diff_predright - diff_predright.min()) * (1/(diff_predright.max() - diff_predright.min()) * 255))
            wandb.log({f"{phase}_diff-pred-right": wandb.Image(diff_predright, caption= f"Epoch {epoch} Predicted Difference with Right Image")})
            wandb.log({f"{phase}_diff-pred-right-scaled": wandb.Image(diff_predright_scale, caption= f"Epoch {epoch} Predicted Difference with Right Image Scaled")})
            diff_predleft = np.subtract(predicted_image_left, left_image)
            diff_predleft_scale = ((diff_predleft - diff_predleft.min()) * (1/(diff_predleft.max() - diff_predleft.min()) * 255))
            wandb.log({f"{phase}_diff-pred-left-scaled": wandb.Image(diff_predleft_scale, caption= f"Epoch {epoch} Predicted Difference with Left Image")})
            diff_leftright = np.subtract(left_image, right_image)
            wandb.log({f"{phase}_diff-left-right": wandb.Image(diff_leftright, caption= f"Epoch {epoch} Difference Left and Right Image")})


def rendering_depth(left_images,right_images,outputs,depth_gt,depth_outputs,warp_identity,cfg,cfgdir,phase,type):
    
    if cfg.settings.rendering==True:
        left_image = left_images.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width+ 2*cfgdir.files.image_pad).detach().to(torch.device('cpu')).numpy()[0]
        left_image = left_image.transpose(1,2,0)[:, cfgdir.files.image_pad:-cfgdir.files.image_pad,:]*255
        wandb.log({f"{phase}_left-image": wandb.Image(left_image, caption= f"Left Image")})
        
        right_image = right_images.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width+ 2*cfgdir.files.image_pad).detach().to(torch.device('cpu')).numpy()[0]
        right_image = right_image.transpose(1,2,0)[:, cfgdir.files.image_pad:-cfgdir.files.image_pad,:]*255        
        wandb.log({f"{phase}_right-image": wandb.Image(right_image, caption= f"Right Image")})

        if warp_identity == 1:    
            predicted_image = outputs.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width).detach().to(torch.device('cpu')).numpy()[0]
            predicted_image = predicted_image.transpose(1,2,0)*255      
            wandb.log({f"{phase}_left-image-predicted": wandb.Image(predicted_image, caption= f"Left Image Predicted")})
            
            # diff_predleft = np.subtract(predicted_image, left_image)
            # wandb.log({f"{phase}_diff-pred-left": wandb.Image(diff_predleft, caption= f"Epoch {epoch} Predicted Difference with Left Image")})
            # diff_leftright = np.subtract(left_image, right_image)
            # wandb.log({f"{phase}_diff-left-right": wandb.Image(diff_leftright, caption= f"Epoch {epoch} Difference Left and Right Image")})

        elif warp_identity == 0:
            predicted_image = outputs.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width).detach().to(torch.device('cpu')).numpy()[0]
            predicted_image = predicted_image.transpose(1,2,0)*255  
            wandb.log({f"{phase}_right-image-predicted": wandb.Image(predicted_image, caption= f"Right Image Predicted")})
            
            # diff_predright = np.subtract(predicted_image, right_image)     
            # wandb.log({f"{phase}_diff-pred-right": wandb.Image(diff_predright, caption= f"Epoch {epoch} Predicted Difference with Right Image")})
            # diff_leftright = np.subtract(left_image, right_image)
            # wandb.log({f"{phase}_diff-left-right": wandb.Image(diff_leftright, caption= f"Epoch {epoch} Difference Left and Right Image")})


        plt.figure()
        plt.imshow(left_image.astype(np.uint8))
        plt.axis('off')
        # plt.colorbar(shrink=0.3)
        plt.tight_layout()
        plt.savefig(f'images/{phase}-left_image.pdf', bbox_inches='tight',pad_inches= 0)
        plt.close()


        # plt.figure()
        # plt.imshow(right_image.astype(np.uint8))
        # plt.axis('off')
        # # plt.colorbar(shrink=0.3)
        # plt.tight_layout()
        # plt.savefig(f'images/{phase}-right_image.pdf', bbox_inches='tight',pad_inches= 0)
        # plt.close()


        # plt.figure()
        # plt.imshow(predicted_image.astype(np.uint8))
        # plt.axis('off')
        # # plt.colorbar(shrink=0.3)
        # plt.tight_layout()
        # plt.savefig(f'images/{phase}-predicted_image-{type}.pdf', bbox_inches='tight',pad_inches= 0)
        # plt.close()

        # # Create a new figure with the required size
        # vmax = 20
        # # depth_image = depth_gt.view(cfgdir.hyperparams.batch_fit, 375, 1242).detach().to(torch.device('cpu')).numpy()[0]
        # # y, x = np.where(depth_image != 0)
        # # z = depth_image[y, x]
        # # plt.figure(figsize=fig_size)
        # # plt.scatter(x, y, s=5, c=z, cmap='magma', vmin=0, vmax=vmax)
        # # plt.gca().invert_yaxis()  # invert the y-axis to match the image coordinate system
        # # plt.axis('off')
        # # plt.tight_layout()
        # # plt.savefig('depth_image.pdf', bbox_inches='tight',pad_inches= 0)
        # # plt.close()

        # predicted_image = depth_outputs.view(cfgdir.hyperparams.batch_fit, 375, 1242).detach().to(torch.device('cpu')).numpy()[0]

        # plt.figure()
        # plt.imshow(predicted_image, cmap='magma', vmin=0, vmax=vmax)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig(f'images/{phase}-predicted_depth_image-{type}.pdf', bbox_inches='tight',pad_inches= 0)
        # plt.close()


def rendering_test(left_images,right_images,outputs,n_layer,epoch,cfg,cfgdir,phase):

    if cfg.settings.rendering==True:
        left_image_norm = left_images.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width_padded).detach().to(torch.device('cpu')).numpy()[0]
        left_image = np.array([left_image_norm[i] * cfgdir.hyperparams.std[i] + cfgdir.hyperparams.mean[i] for i in range(3)]).transpose(1,2,0)[:, 64:-64,:]*255
        wandb.log({f"{phase}_left-image": wandb.Image(left_image, caption= f"Epoch {epoch} Left Image")})
        
        right_image_norm = right_images.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width_padded).detach().to(torch.device('cpu')).numpy()[0]
        right_image = np.array([right_image_norm[i] * cfgdir.hyperparams.std[i] + cfgdir.hyperparams.mean[i] for i in range(3)]).transpose(1,2,0)[:, 64:-64,:]*255         
        wandb.log({f"{phase}_right-image": wandb.Image(right_image, caption= f"Epoch {epoch} Right Image")})

        if n_layer == 0:
            predicted_image = outputs.view(cfgdir.hyperparams.batch_fit, 48, 192, 96).detach().to(torch.device('cpu')).numpy()[0]
        if n_layer == 1:
            predicted_image = outputs.view(cfgdir.hyperparams.batch_fit, 24, 96, 192).detach().to(torch.device('cpu')).numpy()[0]
        if n_layer == 2:
            predicted_image = outputs.view(cfgdir.hyperparams.batch_fit, 12, 48, 384).detach().to(torch.device('cpu')).numpy()[0]
        if n_layer == 3:
            predicted_image = outputs.view(cfgdir.hyperparams.batch_fit, 6, 24, 768).detach().to(torch.device('cpu')).numpy()[0]
        vmax = np.percentile(predicted_image, 99)
        vmin = np.percentile(predicted_image, 1)
        for i in range(10):
            plt.figure()
            plt.imshow(predicted_image[:,:,i], cmap='magma', vmax=vmax, vmin=vmin)
            plt.axis('off')
            plt.savefig(f'layer-{n_layer}-{i}_image.png', bbox_inches='tight',pad_inches= 0)
            wandb.log({f"{phase}_layer-{n_layer}-{i}": wandb.Image(f'layer-{n_layer}-{i}_image.png', caption= f"Epoch {epoch} Layer {n_layer}")})
            plt.close()

def get_testing_batches(outputs,n_layer,cfg,cfgdir,total_classes,total_res_pred):
    if cfg.settings.rendering==True:
        if n_layer == 0:
            predicted_images = outputs.view(cfgdir.hyperparams.batch_fit, 48, 192, 96).detach().to(torch.device('cpu')).numpy()
            predicted_images_norm = (predicted_images - predicted_images.min())/(predicted_images.max() - predicted_images.min())

            classes = []
            left_side = np.reshape(predicted_images_norm[:,:,:16], (-1,96))
            res_pred = np.array(left_side)
            classes.extend(np.size(left_side,0)*['Left Side'])
            right_side = np.reshape(predicted_images_norm[:,:,-16:], (-1,96))
            res_pred = np.append(res_pred, right_side,axis=0)
            classes.extend(np.size(right_side,0)*['Right Side'])
            cllb = np.reshape(predicted_images_norm[:,:24,16:56], (-1,96))
            res_pred = np.append(res_pred, cllb,axis=0)
            classes.extend(np.size(cllb,0)*['CLLB'])
            cllt = np.reshape(predicted_images_norm[:,24:,16:56], (-1,96))
            res_pred = np.append(res_pred, cllt,axis=0)
            classes.extend(np.size(cllt,0)*['CLLT'])
            clb = np.reshape(predicted_images_norm[:,:24,56:96], (-1,96))
            res_pred = np.append(res_pred, clb,axis=0)
            classes.extend(np.size(clb,0)*['CLB'])
            clt = np.reshape(predicted_images_norm[:,24:,56:96], (-1,96))
            res_pred = np.append(res_pred, clt,axis=0)
            classes.extend(np.size(clt,0)*['CLT'])
            crb = np.reshape(predicted_images_norm[:,:24,96:136], (-1,96))
            res_pred = np.append(res_pred, crb,axis=0)
            classes.extend(np.size(crb,0)*['CRB'])
            crt = np.reshape(predicted_images_norm[:,24:,96:136], (-1,96))
            res_pred = np.append(res_pred, crt,axis=0)
            classes.extend(np.size(crt,0)*['CRT'])
            crrb = np.reshape(predicted_images_norm[:,:24,136:176], (-1,96))
            res_pred = np.append(res_pred, crrb,axis=0)
            classes.extend(np.size(crrb,0)*['CRRB'])
            crrt = np.reshape(predicted_images_norm[:,24:,136:176], (-1,96))
            res_pred = np.append(res_pred, crrt,axis=0)
            classes.extend(np.size(crrt,0)*['CRRT'])

            total_res_pred.extend(res_pred)
            total_classes.extend(classes)
    return total_classes, total_res_pred

def rendering_scatter(outputs,n_layer,epoch,cfg,cfgdir,phase):
    
    if cfg.settings.rendering==True:
        pca = sklearnPCA(n_components=2)
        lda = LDA(n_components=2)
        
        if n_layer == 0:
            predicted_images = outputs.view(cfgdir.hyperparams.batch_fit, 48, 192, 96).detach().to(torch.device('cpu')).numpy()
            predicted_images_norm = (predicted_images - predicted_images.min())/(predicted_images.max() - predicted_images.min())

            classes = []
            left_side = np.reshape(predicted_images_norm[:,:,:16], (-1,96))
            res_pred = np.array(left_side)
            classes.extend(np.size(left_side,0)*['Left Side'])
            right_side = np.reshape(predicted_images_norm[:,:,-16:], (-1,96))
            res_pred = np.append(res_pred, right_side,axis=0)
            classes.extend(np.size(right_side,0)*['Right Side'])
            cllb = np.reshape(predicted_images_norm[:,:24,16:56], (-1,96))
            res_pred = np.append(res_pred, cllb,axis=0)
            classes.extend(np.size(cllb,0)*['CLLB'])
            cllt = np.reshape(predicted_images_norm[:,24:,16:56], (-1,96))
            res_pred = np.append(res_pred, cllt,axis=0)
            classes.extend(np.size(cllt,0)*['CLLT'])
            clb = np.reshape(predicted_images_norm[:,:24,56:96], (-1,96))
            res_pred = np.append(res_pred, clb,axis=0)
            classes.extend(np.size(clb,0)*['CLB'])
            clt = np.reshape(predicted_images_norm[:,24:,56:96], (-1,96))
            res_pred = np.append(res_pred, clt,axis=0)
            classes.extend(np.size(clt,0)*['CLT'])
            crb = np.reshape(predicted_images_norm[:,:24,96:136], (-1,96))
            res_pred = np.append(res_pred, crb,axis=0)
            classes.extend(np.size(crb,0)*['CRB'])
            crt = np.reshape(predicted_images_norm[:,24:,96:136], (-1,96))
            res_pred = np.append(res_pred, crt,axis=0)
            classes.extend(np.size(crt,0)*['CRT'])
            crrb = np.reshape(predicted_images_norm[:,:24,136:176], (-1,96))
            res_pred = np.append(res_pred, crrb,axis=0)
            classes.extend(np.size(crrb,0)*['CRRB'])
            crrt = np.reshape(predicted_images_norm[:,24:,136:176], (-1,96))
            res_pred = np.append(res_pred, crrt,axis=0)
            classes.extend(np.size(crrt,0)*['CRRT'])
 
            # res_pred = pd.DataFrame(lda.fit_transform(res_pred, classes))
            res_pred = pd.DataFrame(pca.fit_transform(res_pred))
            classes = pd.DataFrame((classes), columns=['Class'])
            dataframe = pd.concat([classes,res_pred], axis=1)
        if n_layer == 1:

            predicted_images = outputs.view(cfgdir.hyperparams.batch_fit, 24, 96, 192).detach().to(torch.device('cpu')).numpy()
            predicted_images_norm = (predicted_images - predicted_images.min())/(predicted_images.max() - predicted_images.min())

            classes = []
            left_side = np.reshape(predicted_images_norm[:,:,:8], (-1,192))
            res_pred = np.array(left_side)
            classes.extend(np.size(left_side,0)*['Left Side'])
            right_side = np.reshape(predicted_images_norm[:,:,-8:], (-1,192))
            res_pred = np.append(res_pred, right_side,axis=0)
            classes.extend(np.size(right_side,0)*['Right Side'])
            cllb = np.reshape(predicted_images_norm[:,:12,8:28], (-1,192))
            res_pred = np.append(res_pred, cllb,axis=0)
            classes.extend(np.size(cllb,0)*['CLLB'])
            cllt = np.reshape(predicted_images_norm[:,12:,8:28], (-1,192))
            res_pred = np.append(res_pred, cllt,axis=0)
            classes.extend(np.size(cllt,0)*['CLLT'])
            clb = np.reshape(predicted_images_norm[:,:12,28:48], (-1,192))
            res_pred = np.append(res_pred, clb,axis=0)
            classes.extend(np.size(clb,0)*['CLB'])
            clt = np.reshape(predicted_images_norm[:,12:,28:48], (-1,192))
            res_pred = np.append(res_pred, clt,axis=0)
            classes.extend(np.size(clt,0)*['CLT'])
            crb = np.reshape(predicted_images_norm[:,:12,48:68], (-1,192))
            res_pred = np.append(res_pred, crb,axis=0)
            classes.extend(np.size(crb,0)*['CRB'])
            crt = np.reshape(predicted_images_norm[:,12:,48:68], (-1,192))
            res_pred = np.append(res_pred, crt,axis=0)
            classes.extend(np.size(crt,0)*['CRT'])
            crrb = np.reshape(predicted_images_norm[:,:12,68:88], (-1,192))
            res_pred = np.append(res_pred, crrb,axis=0)
            classes.extend(np.size(crrb,0)*['CRRB'])
            crrt = np.reshape(predicted_images_norm[:,12:,68:88], (-1,192))
            res_pred = np.append(res_pred, crrt,axis=0)
            classes.extend(np.size(crrt,0)*['CRRT'])
 
            res_pred = pd.DataFrame(pca.fit_transform(res_pred))
            classes = pd.DataFrame((classes), columns=['Class'])
            dataframe = pd.concat([classes,res_pred], axis=1)

        if n_layer == 2:
            predicted_images = outputs.view(cfgdir.hyperparams.batch_fit, 12, 48, 384).detach().to(torch.device('cpu')).numpy()
            predicted_images_norm = (predicted_images - predicted_images.min())/(predicted_images.max() - predicted_images.min())

            classes = []
            left_side = np.reshape(predicted_images_norm[:,:,:4], (-1,384))
            res_pred = np.array(left_side)
            classes.extend(np.size(left_side,0)*['Left Side'])
            right_side = np.reshape(predicted_images_norm[:,:,-4:], (-1,384))
            res_pred = np.append(res_pred, right_side,axis=0)
            classes.extend(np.size(right_side,0)*['Right Side'])
            cllb = np.reshape(predicted_images_norm[:,:6,4:14], (-1,384))
            res_pred = np.append(res_pred, cllb,axis=0)
            classes.extend(np.size(cllb,0)*['CLLB'])
            cllt = np.reshape(predicted_images_norm[:,6:,4:14], (-1,384))
            res_pred = np.append(res_pred, cllt,axis=0)
            classes.extend(np.size(cllt,0)*['CLLT'])
            clb = np.reshape(predicted_images_norm[:,:6,14:24], (-1,384))
            res_pred = np.append(res_pred, clb,axis=0)
            classes.extend(np.size(clb,0)*['CLB'])
            clt = np.reshape(predicted_images_norm[:,6:,14:24], (-1,384))
            res_pred = np.append(res_pred, clt,axis=0)
            classes.extend(np.size(clt,0)*['CLT'])
            crb = np.reshape(predicted_images_norm[:,:6,24:34], (-1,384))
            res_pred = np.append(res_pred, crb,axis=0)
            classes.extend(np.size(crb,0)*['CRB'])
            crt = np.reshape(predicted_images_norm[:,6:,24:34], (-1,384))
            res_pred = np.append(res_pred, crt,axis=0)
            classes.extend(np.size(crt,0)*['CRT'])
            crrb = np.reshape(predicted_images_norm[:,:6,34:44], (-1,384))
            res_pred = np.append(res_pred, crrb,axis=0)
            classes.extend(np.size(crrb,0)*['CRRB'])
            crrt = np.reshape(predicted_images_norm[:,6:,34:44], (-1,384))
            res_pred = np.append(res_pred, crrt,axis=0)
            classes.extend(np.size(crrt,0)*['CRRT'])
 
            res_pred = pd.DataFrame(pca.fit_transform(res_pred))
            classes = pd.DataFrame((classes), columns=['Class'])
            dataframe = pd.concat([classes,res_pred], axis=1)
        if n_layer == 3:
            predicted_images = outputs.view(cfgdir.hyperparams.batch_fit, 6, 24, 768).detach().to(torch.device('cpu')).numpy()
            predicted_images_norm = (predicted_images - predicted_images.min())/(predicted_images.max() - predicted_images.min())

            classes = []
            left_side = np.reshape(predicted_images_norm[:,:,:2], (-1,768))
            res_pred = np.array(left_side)
            classes.extend(np.size(left_side,0)*['Left Side'])
            right_side = np.reshape(predicted_images_norm[:,:,-2:], (-1,768))
            res_pred = np.append(res_pred, right_side,axis=0)
            classes.extend(np.size(right_side,0)*['Right Side'])
            cllb = np.reshape(predicted_images_norm[:,:3,2:7], (-1,768))
            res_pred = np.append(res_pred, cllb,axis=0)
            classes.extend(np.size(cllb,0)*['CLLB'])
            cllt = np.reshape(predicted_images_norm[:,3:,2:7], (-1,768))
            res_pred = np.append(res_pred, cllt,axis=0)
            classes.extend(np.size(cllt,0)*['CLLT'])
            clb = np.reshape(predicted_images_norm[:,:3,7:12], (-1,768))
            res_pred = np.append(res_pred, clb,axis=0)
            classes.extend(np.size(clb,0)*['CLB'])
            clt = np.reshape(predicted_images_norm[:,3:,7:12], (-1,768))
            res_pred = np.append(res_pred, clt,axis=0)
            classes.extend(np.size(clt,0)*['CLT'])
            crb = np.reshape(predicted_images_norm[:,:3,12:17], (-1,768))
            res_pred = np.append(res_pred, crb,axis=0)
            classes.extend(np.size(crb,0)*['CRB'])
            crt = np.reshape(predicted_images_norm[:,3:,12:17], (-1,768))
            res_pred = np.append(res_pred, crt,axis=0)
            classes.extend(np.size(crt,0)*['CRT'])
            crrb = np.reshape(predicted_images_norm[:,:3,17:22], (-1,768))
            res_pred = np.append(res_pred, crrb,axis=0)
            classes.extend(np.size(crrb,0)*['CRRB'])
            crrt = np.reshape(predicted_images_norm[:,3:,17:22], (-1,768))
            res_pred = np.append(res_pred, crrt,axis=0)
            classes.extend(np.size(crrt,0)*['CRRT'])
 
            res_pred = pd.DataFrame(pca.fit_transform(res_pred))
            classes = pd.DataFrame((classes), columns=['Class'])
            dataframe = pd.concat([classes,res_pred], axis=1)

        fig, axes = plt.subplots(2,5, figsize=(20,8))
        
        class_names = ['Left Side','CLLT','CLT','CRT','CRRT','Right Side', 'CLLB','CLB','CRB','CRRB']#list(dataframe['Class'].unique())

        for i, ax in enumerate(axes.flat):
            sns.set_theme()
            try:
                sns.kdeplot(data = dataframe.loc[dataframe['Class']==class_names[i]], x=0, y=1, levels=5, thresh=.2, ax=ax)
            except:
                sns.scatterplot(data = dataframe.loc[dataframe['Class']==class_names[i]], x=0, y=1, ax=ax)
            
            ax.set_title(f'Class: {class_names[i]}')
            # ax.set_xlim(-1,1)
            # ax.set_ylim(-1,1)
            ax.set_xlabel('')
            ax.set_ylabel('')

        
        fig.savefig(f'multi_image.png', bbox_inches='tight',pad_inches= 0)
        wandb.log({f"multi_image": wandb.Image(f'multi_image.png', caption= f"Epoch {epoch} Multi Image")})


def rendering_testing_scatter(classes,res_pred,epoch,cfg):
    
    if cfg.settings.rendering==True:
        pca = sklearnPCA(n_components=2)
        lda = LDA(n_components=2)
        print(np.shape(classes))
        print(np.shape(res_pred))
 
        res_pred = pd.DataFrame(pca.fit_transform(res_pred))
        classes = pd.DataFrame((classes), columns=['Class'])
        dataframe = pd.concat([classes,res_pred], axis=1)

        fig, axes = plt.subplots(2,5, figsize=(20,8))
        
        class_names = ['Left Side','CLLT','CLT','CRT','CRRT','Right Side', 'CLLB','CLB','CRB','CRRB']#list(dataframe['Class'].unique())

        for i, ax in enumerate(axes.flat):
            sns.set_theme()
            try:
                sns.kdeplot(data = dataframe.loc[dataframe['Class']==class_names[i]], x=0, y=1, levels=5, thresh=.2, ax=ax)
            except:
                sns.scatterplot(data = dataframe.loc[dataframe['Class']==class_names[i]], x=0, y=1, ax=ax)
            
            ax.set_title(f'Class: {class_names[i]}')
            # ax.set_xlim(-1,1)
            # ax.set_ylim(-1,1)
            ax.set_xlabel('')
            ax.set_ylabel('')

        
        fig.savefig(f'multi_image.png', bbox_inches='tight',pad_inches= 0)
        wandb.log({f"multi_image": wandb.Image(f'multi_image.png', caption= f"Epoch {epoch} Multi Image")})

def rendering_disparities(left_images,right_images,left_images_prediction,right_images_prediction,cfgdir):
    max_disparity=64
    csize=(7,7)
    bsize=(3,3)
    P1 = 5
    P2 = 70

    paths = Paths()
    left_image = left_images.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width+ 2*cfgdir.files.image_pad).detach().to(torch.device('cpu')).numpy()[0]
    left_image = left_image.transpose(1,2,0)[:, cfgdir.files.image_pad:-cfgdir.files.image_pad,:]*255
    wandb.log({f"left-image": wandb.Image(left_image, caption= f"Left Image")})
    left_image = cv2.GaussianBlur(cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY), bsize, 0, 0)
    

    right_image = right_images.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width+ 2*cfgdir.files.image_pad).detach().to(torch.device('cpu')).numpy()[0]
    right_image = right_image.transpose(1,2,0)[:, cfgdir.files.image_pad:-cfgdir.files.image_pad,:]*255        
    wandb.log({f"right-image": wandb.Image(right_image, caption= f"Right Image")})
    right_image = cv2.GaussianBlur(cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY), bsize, 0, 0)

    left_cost_volume, right_cost_volume = compute_costs(left_image, right_image, max_disparity, csize)

    left_aggregation_volume = aggregate_costs(left_cost_volume, P1, P2, paths)
    right_aggregation_volume = aggregate_costs(right_cost_volume, P1, P2, paths)

    left_disparity_map = np.uint8(select_disparity(left_aggregation_volume))
    right_disparity_map = np.uint8(select_disparity(right_aggregation_volume))


    left_disparity_map = cv2.medianBlur(left_disparity_map, bsize[0])
    right_disparity_map = cv2.medianBlur(right_disparity_map, bsize[0])

    vmax = np.percentile(left_disparity_map, 99)  
    vmin = -10

    plt.figure()
    plt.imshow(left_disparity_map, cmap='magma', vmax=vmax, vmin=vmin)
    plt.axis('off')
    plt.colorbar(shrink=0.3)
    plt.tight_layout()
    plt.savefig(f'left_disparity.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
    wandb.log({f"left-disparity": wandb.Image(f'left_disparity.png', caption= f"Left Disparity GT")})
    plt.close()

    plt.figure()
    plt.imshow(right_disparity_map, cmap='magma', vmax=vmax, vmin=vmin)
    plt.axis('off')
    plt.colorbar(shrink=0.3)
    plt.tight_layout()
    plt.savefig(f'right_disparity.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
    wandb.log({f"right-disparity": wandb.Image(f'right_disparity.png', caption= f"Right Disparity GT")})
    plt.close()


    for i in range(len(left_images_prediction)):

        left_image_prediction = left_images_prediction[i].view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width+ 2*cfgdir.files.image_pad).detach().to(torch.device('cpu')).numpy()[0]
        left_image_prediction = left_image_prediction.transpose(1,2,0)[:, cfgdir.files.image_pad:-cfgdir.files.image_pad,:]*255
        wandb.log({f"left-image-prediction-{i}": wandb.Image(left_image_prediction, caption= f"Left Image Prediction {i}")})
        left_image_prediction = cv2.GaussianBlur(cv2.cvtColor(left_image_prediction, cv2.COLOR_BGR2GRAY), bsize, 0, 0)
        
        right_image_prediction = right_images_prediction[i].view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width+ 2*cfgdir.files.image_pad).detach().to(torch.device('cpu')).numpy()[0]
        right_image_prediction = right_image_prediction.transpose(1,2,0)[:, cfgdir.files.image_pad:-cfgdir.files.image_pad,:]*255
        wandb.log({f"right-image-prediction-{i}": wandb.Image(right_image_prediction, caption= f"Right Image Prediction {i}")})
        right_image_prediction = cv2.GaussianBlur(cv2.cvtColor(right_image_prediction, cv2.COLOR_BGR2GRAY), bsize, 0, 0)


        # if i == len(left_images_prediction) - 1:
        #     wandb.log({f"left-image-prediction-bw-{i}": wandb.Image(left_image_prediction, caption= f"Left Image Prediction BW {i}")})
        #     wandb.log({f"right-image-prediction-bw-{i}": wandb.Image(right_image_prediction, caption= f"Right Image Prediction BW {i}")})

        #     # vmax = 325
        #     # vmin = 0
        #     plt.figure()
        #     plt.imshow(left_image_prediction, cmap='magma_r', vmax=vmax, vmin=vmin)
        #     plt.axis('off')
        #     plt.colorbar(shrink=0.3)
        #     plt.tight_layout()
        #     plt.savefig(f'left_disparity.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
        #     wandb.log({f"left-image_prediction-enc-{i}": wandb.Image(f'left_disparity.png', caption= f"Left Imaage Prediction Mag {i}")})
        #     plt.close()

        #     plt.figure()
        #     plt.imshow(right_image_prediction, cmap='magma_r', vmax=vmax, vmin=vmin)
        #     plt.axis('off')
        #     plt.colorbar(shrink=0.3)
        #     plt.tight_layout()
        #     plt.savefig(f'right_disparity.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
        #     wandb.log({f"right-image_prediction-enc-{i}": wandb.Image(f'right_disparity.png', caption= f"Right Image Prediction Mag {i}")})
        #     plt.close()

        left_cost_volume_rp, right_cost_volume_rp = compute_costs(left_image, right_image_prediction, max_disparity, csize)
        left_cost_volume_lp, right_cost_volume_lp = compute_costs(left_image_prediction, right_image, max_disparity, csize)
        left_cost_volume_p, right_cost_volume_p = compute_costs(left_image_prediction, right_image_prediction, max_disparity, csize)

        left_aggregation_volume_rp = aggregate_costs(left_cost_volume_rp, P1, P2, paths)
        right_aggregation_volume_rp = aggregate_costs(right_cost_volume_rp, P1, P2, paths)
        left_aggregation_volume_lp = aggregate_costs(left_cost_volume_lp, P1, P2, paths)
        right_aggregation_volume_lp = aggregate_costs(right_cost_volume_lp, P1, P2, paths)
        left_aggregation_volume_p = aggregate_costs(left_cost_volume_p, P1, P2, paths)
        right_aggregation_volume_p = aggregate_costs(right_cost_volume_p, P1, P2, paths)

        left_disparity_map_rp = np.uint8(select_disparity(left_aggregation_volume_rp))
        right_disparity_map_rp = np.uint8(select_disparity(right_aggregation_volume_rp))
        left_disparity_map_lp = np.uint8(select_disparity(left_aggregation_volume_lp))
        right_disparity_map_lp = np.uint8(select_disparity(right_aggregation_volume_lp))
        left_disparity_map_p = np.uint8(select_disparity(left_aggregation_volume_p))
        right_disparity_map_p = np.uint8(select_disparity(right_aggregation_volume_p))


        left_disparity_map_rp = cv2.medianBlur(left_disparity_map_rp, bsize[0])
        right_disparity_map_rp = cv2.medianBlur(right_disparity_map_rp, bsize[0])
        left_disparity_map_lp = cv2.medianBlur(left_disparity_map_lp, bsize[0])
        right_disparity_map_lp = cv2.medianBlur(right_disparity_map_lp, bsize[0])
        left_disparity_map_p = cv2.medianBlur(left_disparity_map_p, bsize[0])
        right_disparity_map_p = cv2.medianBlur(right_disparity_map_p, bsize[0])

        # vmax = np.percentile(left_disparity_map_rp, 99)  
        # vmin = -10

        plt.figure()
        plt.imshow(left_disparity_map_rp, cmap='magma', vmax=vmax, vmin=vmin)
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.tight_layout()
        plt.savefig(f'left_disparity_rp.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
        wandb.log({f"left-disparity-rp-{i}": wandb.Image(f'left_disparity_rp.png', caption= f"Left Disparity Right Predicted {i}")})
        plt.close()

        plt.figure()
        plt.imshow(right_disparity_map_rp, cmap='magma', vmax=vmax, vmin=vmin)
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.tight_layout()
        plt.savefig(f'right_disparity_rp.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
        wandb.log({f"right-disparity-rp-{i}": wandb.Image(f'right_disparity_rp.png', caption= f"Right Disparity Right Predicted {i}")})
        plt.close()


        plt.figure()
        plt.imshow(left_disparity_map_lp, cmap='magma', vmax=vmax, vmin=vmin)
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.tight_layout()
        plt.savefig(f'left_disparity_lp.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
        wandb.log({f"left-disparity-lp-{i}": wandb.Image(f'left_disparity_lp.png', caption= f"Left Disparity Left Predicted {i}")})
        plt.close()

        plt.figure()
        plt.imshow(right_disparity_map_lp, cmap='magma', vmax=vmax, vmin=vmin)
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.tight_layout()
        plt.savefig(f'right_disparity_lp.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
        wandb.log({f"right-disparity-lp-{i}": wandb.Image(f'right_disparity_lp.png', caption= f"Right Disparity Left Predicted {i}")})
        plt.close()      

        plt.figure()
        plt.imshow(left_disparity_map_p, cmap='magma', vmax=vmax, vmin=vmin)
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.tight_layout()
        plt.savefig(f'left_disparity_p.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
        wandb.log({f"left-disparity-p-{i}": wandb.Image(f'left_disparity_p.png', caption= f"Left Disparity All Predicted {i}")})
        plt.close()

        plt.figure()
        plt.imshow(right_disparity_map_p, cmap='magma', vmax=vmax, vmin=vmin)
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.tight_layout()
        plt.savefig(f'right_disparity_p.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
        wandb.log({f"right-disparity-p-{i}": wandb.Image(f'right_disparity_p.png', caption= f"Right Disparity All Predicted {i}")})
        plt.close()  

def rendering_disparities_layered(left_images,right_images,left_images_prediction,right_images_prediction,cfgdir):
    max_disparity=64
    csize=(7,7)
    bsize=(3,3)
    P1 = 10
    P2 = 120

    paths = Paths()
    left_image = left_images.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width+ 2*cfgdir.files.image_pad).detach().to(torch.device('cpu')).numpy()[0]
    left_image = left_image.transpose(1,2,0)[:, cfgdir.files.image_pad:,:]*255
    wandb.log({f"left-image": wandb.Image(left_image, caption= f"Left Image")})
    left_image = cv2.GaussianBlur(cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY), bsize, 0, 0)
    

    right_image = right_images.view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width+ 2*cfgdir.files.image_pad).detach().to(torch.device('cpu')).numpy()[0]
    right_image = right_image.transpose(1,2,0)[:, cfgdir.files.image_pad:,:]*255        
    wandb.log({f"right-image": wandb.Image(right_image, caption= f"Right Image")})
    right_image = cv2.GaussianBlur(cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY), bsize, 0, 0)

    left_cost_volume, right_cost_volume = compute_costs(left_image, right_image, max_disparity, csize)

    left_aggregation_volume = aggregate_costs(left_cost_volume, P1, P2, paths)
    right_aggregation_volume = aggregate_costs(right_cost_volume, P1, P2, paths)

    left_disparity_map = np.uint8(select_disparity(left_aggregation_volume))
    right_disparity_map = np.uint8(select_disparity(right_aggregation_volume))


    left_disparity_map = cv2.medianBlur(left_disparity_map, bsize[0])
    right_disparity_map = cv2.medianBlur(right_disparity_map, bsize[0])

    vmax = np.percentile(left_disparity_map, 99)  
    vmin = -10

    plt.figure()
    plt.imshow(left_disparity_map, cmap='magma', vmax=vmax, vmin=vmin)
    plt.axis('off')
    plt.colorbar(shrink=0.3)
    plt.tight_layout()
    plt.savefig(f'left_disparity.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
    wandb.log({f"left-disparity": wandb.Image(f'left_disparity.png', caption= f"Left Disparity GT")})
    plt.close()

    plt.figure()
    plt.imshow(right_disparity_map, cmap='magma', vmax=vmax, vmin=vmin)
    plt.axis('off')
    plt.colorbar(shrink=0.3)
    plt.tight_layout()
    plt.savefig(f'right_disparity.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
    wandb.log({f"right-disparity": wandb.Image(f'right_disparity.png', caption= f"Right Disparity GT")})
    plt.close()


    for i in range(len(left_images_prediction)):

        left_image_prediction = left_images_prediction[i].view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width+ 2*cfgdir.files.image_pad).detach().to(torch.device('cpu')).numpy()[0]
        left_image_prediction = left_image_prediction.transpose(1,2,0)[:, cfgdir.files.image_pad:-cfgdir.files.image_pad,:]*255
        wandb.log({f"left-image-prediction-{i}": wandb.Image(left_image_prediction, caption= f"Left Image Prediction {i}")})
        left_image_prediction = cv2.GaussianBlur(cv2.cvtColor(left_image_prediction, cv2.COLOR_BGR2GRAY), bsize, 0, 0)
        
        right_image_prediction = right_images_prediction[i].view(cfgdir.hyperparams.batch_fit, 3, cfgdir.files.image_height, cfgdir.files.image_width+ 2*cfgdir.files.image_pad).detach().to(torch.device('cpu')).numpy()[0]
        right_image_prediction = right_image_prediction.transpose(1,2,0)[:, cfgdir.files.image_pad:-cfgdir.files.image_pad,:]*255
        wandb.log({f"right-image-prediction-{i}": wandb.Image(right_image_prediction, caption= f"Right Image Prediction {i}")})
        right_image_prediction = cv2.GaussianBlur(cv2.cvtColor(right_image_prediction, cv2.COLOR_BGR2GRAY), bsize, 0, 0)


        # wandb.log({f"left-image-prediction-bw-{i}": wandb.Image(left_image_prediction, caption= f"Left Image Prediction BW {i}")})
        # wandb.log({f"right-image-prediction-bw-{i}": wandb.Image(right_image_prediction, caption= f"Right Image Prediction BW {i}")})

        vmax = int(np.percentile(left_image_prediction, 99)*1.3)
        vmin = np.percentile(left_image_prediction, 1)
        plt.figure()
        plt.imshow(left_image_prediction, cmap='magma_r', vmax=vmax, vmin=vmin)
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.tight_layout()
        plt.savefig(f'left_disparity.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
        wandb.log({f"left-image_prediction-enc-{i}": wandb.Image(f'left_disparity.png', caption= f"Left Imaage Prediction Mag {i}")})
        plt.close()

        plt.figure()
        plt.imshow(right_image_prediction, cmap='magma_r', vmax=vmax, vmin=vmin)
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.tight_layout()
        plt.savefig(f'right_disparity.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
        wandb.log({f"right-image_prediction-enc-{i}": wandb.Image(f'right_disparity.png', caption= f"Right Image Prediction Mag {i}")})
        plt.close()

        if i == 0:

            left_cost_volume_rp, right_cost_volume_rp = compute_costs(left_image, right_image_prediction, max_disparity, csize)
            left_cost_volume_lp, right_cost_volume_lp = compute_costs(left_image_prediction, right_image, max_disparity, csize)
            left_cost_volume_p, right_cost_volume_p = compute_costs(left_image_prediction, right_image_prediction, max_disparity, csize)

            left_aggregation_volume_rp = aggregate_costs(left_cost_volume_rp, P1, P2, paths)
            right_aggregation_volume_rp = aggregate_costs(right_cost_volume_rp, P1, P2, paths)
            left_aggregation_volume_lp = aggregate_costs(left_cost_volume_lp, P1, P2, paths)
            right_aggregation_volume_lp = aggregate_costs(right_cost_volume_lp, P1, P2, paths)
            left_aggregation_volume_p = aggregate_costs(left_cost_volume_p, P1, P2, paths)
            right_aggregation_volume_p = aggregate_costs(right_cost_volume_p, P1, P2, paths)

            left_disparity_map_rp = np.uint8(select_disparity(left_aggregation_volume_rp))
            right_disparity_map_rp = np.uint8(select_disparity(right_aggregation_volume_rp))
            left_disparity_map_lp = np.uint8(select_disparity(left_aggregation_volume_lp))
            right_disparity_map_lp = np.uint8(select_disparity(right_aggregation_volume_lp))
            left_disparity_map_p = np.uint8(select_disparity(left_aggregation_volume_p))
            right_disparity_map_p = np.uint8(select_disparity(right_aggregation_volume_p))


            left_disparity_map_rp = cv2.medianBlur(left_disparity_map_rp, bsize[0])
            right_disparity_map_rp = cv2.medianBlur(right_disparity_map_rp, bsize[0])
            left_disparity_map_lp = cv2.medianBlur(left_disparity_map_lp, bsize[0])
            right_disparity_map_lp = cv2.medianBlur(right_disparity_map_lp, bsize[0])
            left_disparity_map_p = cv2.medianBlur(left_disparity_map_p, bsize[0])
            right_disparity_map_p = cv2.medianBlur(right_disparity_map_p, bsize[0])

            vmax = np.percentile(left_disparity_map, 99)  
            vmin = -10

            plt.figure()
            plt.imshow(left_disparity_map_rp, cmap='magma', vmax=vmax, vmin=vmin)
            plt.axis('off')
            plt.colorbar(shrink=0.3)
            plt.tight_layout()
            plt.savefig(f'left_disparity_rp.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
            wandb.log({f"left-disparity-rp-{i}": wandb.Image(f'left_disparity_rp.png', caption= f"Left Disparity Right Predicted {i}")})
            plt.close()

            plt.figure()
            plt.imshow(right_disparity_map_rp, cmap='magma', vmax=vmax, vmin=vmin)
            plt.axis('off')
            plt.colorbar(shrink=0.3)
            plt.tight_layout()
            plt.savefig(f'right_disparity_rp.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
            wandb.log({f"right-disparity-rp-{i}": wandb.Image(f'right_disparity_rp.png', caption= f"Right Disparity Right Predicted {i}")})
            plt.close()

            plt.figure()
            plt.imshow(left_disparity_map_lp, cmap='magma', vmax=vmax, vmin=vmin)
            plt.axis('off')
            plt.colorbar(shrink=0.3)
            plt.tight_layout()
            plt.savefig(f'left_disparity_lp.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
            wandb.log({f"left-disparity-lp-{i}": wandb.Image(f'left_disparity_lp.png', caption= f"Left Disparity Left Predicted {i}")})
            plt.close()

            plt.figure()
            plt.imshow(right_disparity_map_lp, cmap='magma', vmax=vmax, vmin=vmin)
            plt.axis('off')
            plt.colorbar(shrink=0.3)
            plt.tight_layout()
            plt.savefig(f'right_disparity_lp.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
            wandb.log({f"right-disparity-lp-{i}": wandb.Image(f'right_disparity_lp.png', caption= f"Right Disparity Left Predicted {i}")})
            plt.close()      

            plt.figure()
            plt.imshow(left_disparity_map_p, cmap='magma', vmax=vmax, vmin=vmin)
            plt.axis('off')
            plt.colorbar(shrink=0.3)
            plt.tight_layout()
            plt.savefig(f'left_disparity_p.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
            wandb.log({f"left-disparity-p-{i}": wandb.Image(f'left_disparity_p.png', caption= f"Left Disparity All Predicted {i}")})
            plt.close()

            plt.figure()
            plt.imshow(right_disparity_map_p, cmap='magma', vmax=vmax, vmin=vmin)
            plt.axis('off')
            plt.colorbar(shrink=0.3)
            plt.tight_layout()
            plt.savefig(f'right_disparity_p.png', dpi=1200, bbox_inches='tight',pad_inches= 0)
            wandb.log({f"right-disparity-p-{i}": wandb.Image(f'right_disparity_p.png', caption= f"Right Disparity All Predicted {i}")})
            plt.close()  


def save_model(model,run,epoch,cfg,cfgdir):
    artifact = wandb.Artifact(f'{cfg.settings.models[cfg.settings.model_option]}{cfgdir.files.dataloader}_lr-{cfgdir.hyperparams.base_lr}_bs-{cfgdir.hyperparams.batch_size}_epoch-{epoch}', type='model')
    torch.save(model.state_dict(), f'tmp_model.pth')
    artifact.add_file(f'tmp_model.pth')
    run.log_artifact(artifact)

def save_model_depth(model,run,epoch,cfg,cfgdir):
    artifact = wandb.Artifact(f'Depth_{cfg.settings.models[cfg.settings.model_option]}{cfgdir.files.dataloader}_lr-{cfgdir.hyperparams.base_lr}_bs-{cfgdir.hyperparams.batch_size}_epoch-{epoch}', type='model')
    torch.save(model.state_dict(), f'tmp_model.pth')
    artifact.add_file(f'tmp_model.pth')
    run.log_artifact(artifact)

def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    gt = gt.detach().clone().to('cpu')
    pred = pred.detach().clone().to('cpu')
    pred[pred < 1e-3] = 1e-3
    pred[pred > 80] = 80
    gt[gt < 1e-3] = 1e-3
    gt[gt > 80] = 80
    gt = np.array(gt)
    pred = np.array(pred)

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return  {'a1': a1, 'a2': a2, 'a3': a3, 'rmse': rmse, 'rmse log': rmse_log, 'abs rel': abs_rel, 'sq rel': sq_rel}

def data_graft(left_images, right_images):
    rand_w = random.randint(0, 4) / 5
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



def generate_warp_image(img, K, T, D):
    batch_size, _, height, width = img.shape
    eps = 1e-7
    inv_K = torch.from_numpy(np.linalg.pinv(K.cpu().numpy())).type_as(D)

    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = nn.Parameter(torch.from_numpy(id_coords), requires_grad=False).type_as(D)

    ones = nn.Parameter(torch.ones(batch_size, 1, height * width), requires_grad=False).type_as(D)

    pix_coords = torch.unsqueeze(torch.stack(
        [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
    pix_coords = pix_coords.repeat(batch_size, 1, 1)
    pix_coords = nn.Parameter(torch.cat([pix_coords, ones], 1), requires_grad=False).type_as(D)

    cam_points = torch.matmul(inv_K[:, :3, :3], pix_coords)
    cam_points = D.reshape(batch_size, 1, -1) * cam_points
    cam_points = torch.cat([cam_points, ones], 1)

    P = torch.matmul(K, T)[:, :3, :]

    cam_points = torch.matmul(P, cam_points)

    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + eps)
    pix_coords = pix_coords.view(batch_size, 2, height, width)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= width - 1
    pix_coords[..., 1] /= height - 1
    pix_coords = (pix_coords - 0.5) * 2

    warp_img = torch.nn.functional.grid_sample(img, pix_coords, padding_mode="border")
    return warp_img
