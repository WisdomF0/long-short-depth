import time

import kornia.augmentation as K
import numpy as np
from math import ceil, floor
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from model.layers import point_projection, PadSameConv2d, ConvReLU2, ConvReLU, Upconv, Refine, SSIM, Backprojection
from utils import conditional_flip, filter_state_dict

from utils import parse_config
from .base_models import DepthAugmentation, ResnetEncoder, EfficientNetEncoder, MonoDepthModule, GWCStereoModule, BruteFusion, GWCDModule
from .ccf_modules import *


class LSDepthModel(nn.Module):
    def __init__(self, inv_depth_min_max=(0.33, 0.0025), cv_depth_steps=32, pretrain_mode=False, pretrain_dropout=0.0, pretrain_dropout_mode=0,
                 augmentation=None, use_mono=True, use_stereo=False, use_ssim=True, sfcv_mult_mask=True,
                 simple_mask=False, mask_use_cv=True, mask_use_feats=True, cv_patch_size=3, depth_large_model=False, no_cv=False,
                 freeze_backbone=True, freeze_module=(), checkpoint_location=None, mask_cp_loc=None, depth_cp_loc=None,
                 fusion_type = 'ccf_fusion', input_size=[256, 512], ccf_mid_dim=32, use_img_in_depthnet=True,
                 backbone_type='resnet18'):
        
        super().__init__()
        self.inv_depth_min_max = inv_depth_min_max
        self.cv_depth_steps = cv_depth_steps
        self.use_mono = use_mono
        self.use_stereo = use_stereo
        self.use_ssim = use_ssim
        self.sfcv_mult_mask = sfcv_mult_mask
        self.pretrain_mode = int(pretrain_mode)
        self.pretrain_dropout = pretrain_dropout
        self.pretrain_dropout_mode = pretrain_dropout_mode
        self.augmentation = augmentation
        self.simple_mask = simple_mask
        self.mask_use_cv = mask_use_cv
        self.mask_use_feats = mask_use_feats
        self.cv_patch_size = cv_patch_size
        self.no_cv = no_cv
        self.depth_large_model = depth_large_model
        self.checkpoint_location = checkpoint_location
        self.mask_cp_loc = mask_cp_loc
        self.depth_cp_loc = depth_cp_loc
        self.freeze_module = freeze_module
        self.freeze_backbone = freeze_backbone

        self.fusion_type = fusion_type
        self.input_size = input_size
        self.ccf_mid_dim = ccf_mid_dim
        self.use_img_in_depthnet = use_img_in_depthnet
        self.backbone_type = backbone_type

        assert self.backbone_type in ["resnet18", "efficientnetb5"]

        self.extra_input_dim = 0

        if self.backbone_type == 'resnet18':
            self._feature_extractor = ResnetEncoder(num_layers=18, pretrained=True)
        elif self.backbone_type == 'efficientnetb5':
            self._feature_extractor = EfficientNetEncoder(pretrained=True)

        self.mono_module = MonoDepthModule(extra_input_dim=self.extra_input_dim,
                                           feature_channels=self._feature_extractor.num_ch_enc, 
                                           large_model=self.depth_large_model)
        self.stereo_module = GWCStereoModule()
        

class BruteModel(nn.Module):
    def __init__(self, inv_depth_min_max=(0.33, 0.0025), cv_depth_steps=32, 
                 backbone_type='resnet18', freez_backbone=True, backbone_pretrain=True,
                 gwc_pretrain=None, freez_module=(), device="cuda"):
        super().__init__()
        self.inv_depth_min_max = inv_depth_min_max
        self.cv_depth_steps = cv_depth_steps
        self.freez_backbone = freez_backbone
        self.backbone_pretrain = backbone_pretrain
        self.gwc_pretrain = gwc_pretrain
        self.freez_module = freez_module
        
        self.backbone_type = backbone_type
        assert self.backbone_type in ["resnet18", "efficientnetb5"]

        if self.backbone_type == 'resnet18':
            self._feature_extractor = ResnetEncoder(num_layers=18, pretrained=self.backbone_pretrain)
        elif self.backbone_type == 'efficientnetb5':
            self._feature_extractor = EfficientNetEncoder(pretrained=self.backbone_pretrain)

        self.fusion_model = BruteFusion(depth_steps=self.cv_depth_steps, feature_channels=self._feature_extractor.num_ch_enc)

        if freez_backbone:
            for p in self._feature_extractor.parameters(True):
                p.requires_grad_(False)

        self.cv_module = GWCStereoModule(self.cv_depth_steps * 4, use_concat_volume=True)
        self.st_module = GWCDModule(56, 1)
        self.mono_module = MonoDepthModule()

        if self.gwc_pretrain is not None:
            if isinstance(self.gwc_pretrain, str):
                state_dict = torch.load(self.gwc_pretrain)
                params_dict = {}
                for key, param in state_dict['model'].items():
                    if "feature_extraction" in key:
                        params_dict[key] = param 

                self.cv_module.load_state_dict(params_dict, strict=False) 

        for module_name in self.freez_module:
            module = self.__getattr__(module_name + "_module")
            module.eval()
            for param in module.parameters(True):
                param.requires_grad_(False)

    def forward(self, data_dict):
        leftframe = data_dict['leftframe']
        rightframe = data_dict['rightframe']

        data_dict["inv_depth_min"] = leftframe.new_tensor([self.inv_depth_min_max[0]])
        data_dict["inv_depth_max"] = leftframe.new_tensor([self.inv_depth_min_max[1]])
        data_dict["cv_depth_steps"] = leftframe.new_tensor([self.cv_depth_steps], dtype=torch.int32)

        with torch.no_grad():
            data_dict = self.cv_module(data_dict)
            data_dict = self.st_module(data_dict)

        data_dict["left_features"] = self._feature_extractor(data_dict["leftframe"] + .5)

        data_dict["cost_volume_init"] = data_dict["cost_volume"]

        data_dict = self.mono_module(data_dict)
        data_dict["predicted_inverse_depths_mono"] = [(1-pred) * self.inv_depth_min_max[1] + pred * self.inv_depth_min_max[0]
                                            for pred in data_dict["predicted_inverse_depths_mono"]]
        mono_depth_pred = torch.clamp(1.0 / data_dict["predicted_inverse_depths_mono"][0], min=1e-3, max=80.0).detach()

        b, c, h, w = leftframe.shape

        depths = (1 / torch.linspace(data_dict["inv_depth_max"][0].item(), data_dict["inv_depth_min"][0].item(), data_dict["cv_depth_steps"][0].item(),
                                    device=leftframe.device))
        data_dict["cv_bin_steps"] = depths

        pseudo_mono_cost = self.pseudocost_from_mono(mono_depth_pred, 
                                    depth_hypothesis = data_dict["cv_bin_steps"].view(1, -1, 1, 1).expand(b, -1, h, w).detach()).detach()
        
        b, d, sh, sw = data_dict["cost_volume"].shape
        oversize = [h - sh, w - sw]
        data_dict["cost_volume"] = F.pad(data_dict["cost_volume"], [floor(oversize[1] / 2), ceil(oversize[1] / 2), floor(oversize[0] / 2), ceil(oversize[0] / 2)])

        data_dict["predicted_inverse_depths"] = [
            (1 - pred) * self.inv_depth_min_max[1] + pred * self.inv_depth_min_max[0]
            for pred in self.fusion_model(pseudo_mono_cost, data_dict["cost_volume"], data_dict["left_features"])
        ]

        data_dict["result"] = data_dict["predicted_inverse_depths"][0]
        data_dict["result_mono"] = data_dict["predicted_inverse_depths_mono"][0]

        return data_dict


    def pseudocost_from_mono(self, monodepth, depth_hypothesis):
        abs_depth_diff = torch.abs(monodepth - depth_hypothesis)
        # find the closest depth bin that the monodepth correlate with
        min_diff_index = torch.argmin(abs_depth_diff, dim=1, keepdim=True)
        pseudo_cost = depth_hypothesis.new_zeros(depth_hypothesis.shape)
        ones = depth_hypothesis.new_ones(depth_hypothesis.shape)
        
        pseudo_cost.scatter_(dim = 1, index = min_diff_index, src = ones)
        
        return pseudo_cost

    def find_mincost_depth(self, cost_volume, depth_hypos):
        argmax = torch.argmax(cost_volume, dim=1, keepdim=True)
        mincost_depth = torch.gather(input=depth_hypos, dim=1, index=argmax)
        return mincost_depth
