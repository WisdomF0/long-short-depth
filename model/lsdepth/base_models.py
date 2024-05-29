import time

import kornia.augmentation as K
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from model.layers import *
from utils import conditional_flip, filter_state_dict

from utils import parse_config


class ResnetEncoder(nn.Module):
    def __init__(self, num_layers, pretrained):
        """
        Adapted from monodepth2.resnet_encoder.py
        """
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: torchvision.models.resnet18,
                   34: torchvision.models.resnet34,
                   50: torchvision.models.resnet50,
                   101: torchvision.models.resnet101,
                   152: torchvision.models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features



class EfficientNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetEncoder, self).__init__()
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b5_ap', pretrained=pretrained)
        # Remove last layer
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        self.original_model = basemodel

        self.num_ch_enc = np.array([24, 40, 64, 176, 2048])

    def forward(self, x):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        for ii in range(3):
            x[:,ii,:,:] = (x[:,ii,:,:] - mean[ii]) / std[ii]
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))

        return (features[4], features[5], features[6], features[8], features[11])


class DepthAugmentation(nn.Module):
    def single_apply(self, x):
        return conditional_flip(x, self._flip_conditions, inplace=False)

    def single_revert(self, x):
        return conditional_flip(x, self._flip_conditions, inplace=False)

    def forward(self, data_dict):
        n, c, h, w = data_dict["keyframe"].shape
        self._flip_conditions = torch.rand(n) < .5
        if "cost_volume" in data_dict:
            conditional_flip(data_dict["cost_volume"], self._flip_conditions)
            for sfcv in data_dict["single_frame_cvs"]:
                conditional_flip(sfcv, self._flip_conditions)
        data_dict["keyframe"] = conditional_flip(data_dict["keyframe"], self._flip_conditions, inplace=False)
        if "mvobj_mask" in data_dict:
            conditional_flip(data_dict["mvobj_mask"], self._flip_conditions)

    def revert(self, data_dict):
        if "cost_volume" in data_dict:
            conditional_flip(data_dict["cost_volume"], self._flip_conditions)
            for sfcv in data_dict["single_frame_cvs"]:
                conditional_flip(sfcv, self._flip_conditions)
        if "cv_mask" in data_dict:
            data_dict["cv_mask"] = conditional_flip(data_dict["cv_mask"], self._flip_conditions, inplace=False)
        conditional_flip(data_dict["keyframe"], self._flip_conditions)
        if "predicted_inverse_depths" in data_dict:
            data_dict["predicted_inverse_depths"] = [conditional_flip(pid, self._flip_conditions, inplace=False) for pid in data_dict["predicted_inverse_depths"]]
        if "predicted_probabilities" in data_dict:
            for pp in data_dict["predicted_probabilities"]:
                conditional_flip(pp, self._flip_conditions)
        if "mvobj_mask" in data_dict:
            conditional_flip(data_dict["mvobj_mask"], self._flip_conditions)
        if "mono_pred" in data_dict and data_dict["mono_pred"] is not None:
            data_dict["mono_pred"] = [conditional_flip(mp, self._flip_conditions, inplace=False) for mp in data_dict["mono_pred"]]
        if "stereo_pred" in data_dict and data_dict["stereo_pred"] is not None:
            data_dict["stereo_pred"] = [conditional_flip(sp, self._flip_conditions, inplace=False) for sp in data_dict["stereo_pred"]]
        if "mask" in data_dict and data_dict["mask"] is not None:
            data_dict["mask"] = conditional_flip(data_dict["mask"], self._flip_conditions, inplace=False)
        if "result" in data_dict and data_dict["result"] is not None:
            data_dict["result"] = conditional_flip(data_dict["result"], self._flip_conditions, inplace=False)

        if "predicted_inverse_depths_mono" in data_dict:
            data_dict["predicted_inverse_depths_mono"] = [conditional_flip(pid, self._flip_conditions, inplace=False) for pid in data_dict["predicted_inverse_depths_mono"]]
        if "result_mono" in data_dict and data_dict["result_mono"] is not None:
            data_dict["result_mono"] = conditional_flip(data_dict["result_mono"], self._flip_conditions, inplace=False)


class BruteFusion(nn.Module):
    def __init__(self, depth_steps=32, feature_channels=(64, 64, 128, 256, 512), large_model=False, use_input_img=True) -> None:
        super().__init__()
        self.depth_steps = depth_steps
        self.feat_chns = feature_channels
        self.use_input_img = use_input_img

        self._in_channels = self.depth_steps * 2

        # if use_input_img:
        #     self._in_channels = self.depth_steps * 2 + 3
        # else:
        #     self._in_channels = self.depth_steps * 2

        self._cv_enc_feat_chns = (48, 64, 128, 192, 256) if not large_model else (48, 64, 128, 256, 512)
        self._dec_feat_chns = (256, 128, 64, 48, 32, 24) if not large_model else (512, 256, 128, 64, 32, 24)

        self.enc = nn.ModuleList([
            nn.Sequential(
                ConvReLU2(in_channels=self._in_channels, out_channels=self._cv_enc_feat_chns[0], kernel_size=7),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[0], out_channels=self._cv_enc_feat_chns[0], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[0], out_channels=self._cv_enc_feat_chns[1], kernel_size=7, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[1], out_channels=self._cv_enc_feat_chns[1], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[1], out_channels=self._cv_enc_feat_chns[2], kernel_size=5, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[2], out_channels=self._cv_enc_feat_chns[2], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[2], out_channels=self._cv_enc_feat_chns[3], kernel_size=5, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[3], out_channels=self._cv_enc_feat_chns[3], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[3], out_channels=self._cv_enc_feat_chns[4], kernel_size=3, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[4], out_channels=self._cv_enc_feat_chns[4], kernel_size=3))
        ])

        self.dec = nn.ModuleList([
            # Refine(in_channels=self._cv_enc_feat_chns[4] + self.feat_chns[3], out_channels=self._dec_feat_chns[0]),
            Refine(in_channels=self._cv_enc_feat_chns[4], out_channels=self._dec_feat_chns[0]),
            nn.Sequential(
            Refine(in_channels=self._cv_enc_feat_chns[3] + self.feat_chns[2] + self._dec_feat_chns[0], out_channels=self._dec_feat_chns[1]),
            ConvReLU2(in_channels=self._dec_feat_chns[1], out_channels=self._dec_feat_chns[1], kernel_size=3)),
            nn.Sequential(
            Refine(in_channels=self._cv_enc_feat_chns[2] + self.feat_chns[1] + self._dec_feat_chns[1], out_channels=self._dec_feat_chns[2]),
            ConvReLU2(in_channels=self._dec_feat_chns[2], out_channels=self._dec_feat_chns[2], kernel_size=3)),
            Refine(in_channels=self._cv_enc_feat_chns[1] + self.feat_chns[0] + self._dec_feat_chns[2], out_channels=self._dec_feat_chns[3]),
            nn.Sequential(
            ConvReLU2(in_channels=self._cv_enc_feat_chns[0] + self._dec_feat_chns[3], out_channels=self._dec_feat_chns[4], kernel_size=3),
            PadSameConv2d(kernel_size=3),
            nn.Conv2d(in_channels=self._dec_feat_chns[4], out_channels=self._dec_feat_chns[5], kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1)
            )
        ])

        self.predictors = nn.ModuleList([nn.Sequential(
            PadSameConv2d(kernel_size=3),
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3))
            for channels in self._dec_feat_chns[:3] + self._dec_feat_chns[-1:]])
        
    def forward(self, mono_pseudo_cost, cost_volume, image_features):
        x = torch.cat([mono_pseudo_cost, cost_volume], dim=1)

        cv_feats = []
        for layer in self.enc:
            x = layer(x)
            cv_feats.append(x)

        predictions = []
        for i, layer in enumerate(self.dec):
            if i == 0:
                # x = torch.cat([cv_feats[-1], image_features[-2]], dim=1)
                x = torch.cat([cv_feats[-1]], dim=1)
            elif i == len(self.dec) - 1:
                x = torch.cat([cv_feats[0], x], dim=1)
            else:
                x = torch.cat([cv_feats[-(i + 1)], image_features[-(i + 2)], x], dim=1)
            x = layer(x)
            if i != len(self.dec) - 2:
                predictions = [(self.predict_depth(x, scale=i - (1 if i == len(self.dec) - 1 else 0)))] + predictions

        return predictions
    
    def predict_depth(self, x, scale):
        x = self.predictors[scale](x)
        x = torch.abs(F.tanh(x))
        return x



class MonoDepthModule(nn.Module):
    def __init__(self, extra_input_dim=0, feature_channels=(64, 64, 128, 256, 512), large_model=False) -> None:
        super().__init__()
        self.extra_input_dim = extra_input_dim
        self.feat_chns = feature_channels
        self._in_channels = self.extra_input_dim + 3
        self._cv_enc_feat_chns = (48, 64, 128, 192, 256) if not large_model else (48, 64, 128, 256, 512)
        self._dec_feat_chns = (256, 128, 64, 48, 32, 24) if not large_model else (512, 256, 128, 64, 32, 24)

        self.enc = nn.ModuleList([
            nn.Sequential(
                ConvReLU2(in_channels=self._in_channels, out_channels=self._cv_enc_feat_chns[0], kernel_size=7),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[0], out_channels=self._cv_enc_feat_chns[0], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[0], out_channels=self._cv_enc_feat_chns[1], kernel_size=7, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[1], out_channels=self._cv_enc_feat_chns[1], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[1], out_channels=self._cv_enc_feat_chns[2], kernel_size=5, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[2], out_channels=self._cv_enc_feat_chns[2], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[2], out_channels=self._cv_enc_feat_chns[3], kernel_size=5, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[3], out_channels=self._cv_enc_feat_chns[3], kernel_size=3)),
            nn.Sequential(
                ConvReLU2(in_channels=self._cv_enc_feat_chns[3], out_channels=self._cv_enc_feat_chns[4], kernel_size=3, stride=2),
                ConvReLU2(in_channels=self._cv_enc_feat_chns[4], out_channels=self._cv_enc_feat_chns[4], kernel_size=3))
        ])

        self.dec = nn.ModuleList([
            # Refine(in_channels=self._cv_enc_feat_chns[4] + self.feat_chns[3], out_channels=self._dec_feat_chns[0]),
            Refine(in_channels=self._cv_enc_feat_chns[4], out_channels=self._dec_feat_chns[0]),
            nn.Sequential(
            Refine(in_channels=self._cv_enc_feat_chns[3] + self.feat_chns[2] + self._dec_feat_chns[0], out_channels=self._dec_feat_chns[1]),
            ConvReLU2(in_channels=self._dec_feat_chns[1], out_channels=self._dec_feat_chns[1], kernel_size=3)),
            nn.Sequential(
            Refine(in_channels=self._cv_enc_feat_chns[2] + self.feat_chns[1] + self._dec_feat_chns[1], out_channels=self._dec_feat_chns[2]),
            ConvReLU2(in_channels=self._dec_feat_chns[2], out_channels=self._dec_feat_chns[2], kernel_size=3)),
            Refine(in_channels=self._cv_enc_feat_chns[1] + self.feat_chns[0] + self._dec_feat_chns[2], out_channels=self._dec_feat_chns[3]),
            nn.Sequential(
            ConvReLU2(in_channels=self._cv_enc_feat_chns[0] + self._dec_feat_chns[3], out_channels=self._dec_feat_chns[4], kernel_size=3),
            PadSameConv2d(kernel_size=3),
            nn.Conv2d(in_channels=self._dec_feat_chns[4], out_channels=self._dec_feat_chns[5], kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1)
            )
        ])

        self.predictors = nn.ModuleList([nn.Sequential(
            PadSameConv2d(kernel_size=3),
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3))
            for channels in self._dec_feat_chns[:3] + self._dec_feat_chns[-1:]])

    def forward(self, data_dict, extra_input=None):
        keyframe = data_dict["leftframe"]
        image_features = data_dict["left_features"]

        x = torch.cat([keyframe, extra_input], dim=1) if extra_input!=None else keyframe

        cv_feats = []
        for layer in self.enc:
            x = layer(x)
            cv_feats.append(x)

        predictions = []
        for i, layer in enumerate(self.dec):
            if i == 0:
                x = torch.cat([cv_feats[-1]], dim=1)
            elif i == len(self.dec) - 1:
                x = torch.cat([cv_feats[0], x], dim=1)
            else:
                x = torch.cat([cv_feats[-(i + 1)], image_features[-(i + 2)], x], dim=1)
            x = layer(x)
            if i != len(self.dec) - 2:
                predictions = [(self.predict_depth(x, scale=i - (1 if i == len(self.dec) - 1 else 0)))] + predictions

        data_dict["predicted_inverse_depths_mono"] = predictions

        return data_dict

    def predict_depth(self, x, scale):
        x = self.predictors[scale](x)
        x = torch.abs(F.tanh(x))
        return x    


class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature}
        else:
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6

class GWCStereoModule(nn.Module):
    def __init__(self, maxdisp, num_groups=32, up_frame="leftframe", up_scale=2, up_mode="bilinear", 
                 use_concat_volume=False):
        super(GWCStereoModule, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume
    
        self.num_groups = num_groups

        self.up = SimpleUp(scale=up_scale, mode=up_mode)
        self.up_frame = up_frame

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

    def forward(self, data_dict):
        leftframe = data_dict["leftframe"]
        rightframe = data_dict["rightframe"]

        if self.up_frame == "leftframe":
            leftframe = self.up(leftframe)
        elif self.up_frame == "rightframe":
            rightframe = self.up(rightframe)

        features_left = self.feature_extraction(leftframe)
        features_right = self.feature_extraction(rightframe)

        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume

        data_dict["cv_module_volume"] = volume

        return data_dict
    
class GWCDModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GWCDModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.enc_chns = (64, 128, 256)
        self.dec_chns = (256, 128, 64, 32)
        
        self.encoder = nn.ModuleList([
            nn.Sequential(
                Conv3DReLU(in_channels=self.in_channels, out_channels=self.enc_chns[0], kernel_size=7),
                Conv3DReLU(in_channels=self.enc_chns[0], out_channels=self.enc_chns[0], kernel_size=3)),
            nn.Sequential(
                Conv3DReLU(in_channels=self.enc_chns[0], out_channels=self.enc_chns[1], kernel_size=5, stride=2),
                Conv3DReLU(in_channels=self.enc_chns[1], out_channels=self.enc_chns[1], kernel_size=3)),
            nn.Sequential(
                Conv3DReLU(in_channels=self.enc_chns[1], out_channels=self.enc_chns[2], kernel_size=3, stride=2),
                Conv3DReLU(in_channels=self.enc_chns[2], out_channels=self.enc_chns[2], kernel_size=3)),
        ])
    
        self.decoder = nn.ModuleList([
            # Refine(in_channels=self._cv_enc_feat_chns[4] + self.feat_chns[3], out_channels=self._dec_feat_chns[0]),
            Refine3D(in_channels=self.enc_chns[2], out_channels=self.dec_chns[0]),
            nn.Sequential(
                Refine3D(in_channels=self.enc_chns[1] + self.dec_chns[0], out_channels=self.dec_chns[1]),
                Conv3DReLU(in_channels=self.dec_chns[1], out_channels=self.dec_chns[1], kernel_size=3)
            ),
            nn.Sequential(
                Refine3D(in_channels=self.enc_chns[0] + self.dec_chns[1], out_channels=self.dec_chns[2]),
                Conv3DReLU(in_channels=self.dec_chns[2], out_channels=self.dec_chns[2], kernel_size=3),
                PadSameConv3d(kernel_size=3),
                nn.Conv3d(in_channels=self.dec_chns[2], out_channels=self.dec_chns[3], kernel_size=3),
                nn.LeakyReLU(negative_slope=0.1)
            ),
        ])

        self.predictor = nn.Sequential(
            nn.Conv3d(in_channels=self.dec_chns[3], out_channels=out_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, data_dict):
        x = data_dict["cv_module_volume"]
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)

        for i, layer in enumerate(self.decoder):
            if i == 0:
                x = torch.cat([features[-1]], dim=1)
            else:
                x = torch.cat([features[-(i+1)], x], dim=1)

            x = layer(x)

        data_dict["cost_volume"] = self.predictor(x).squeeze(1)
        return data_dict


class GwcNet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume=False):
        super(GwcNet, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume

        self.num_groups = 40

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred0, pred1, pred2, pred3]

        else:
            cost3 = self.classif3(out3)
            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred3]

def GwcNet_G(d):
    return GwcNet(d, use_concat_volume=False)

def GwcNet_GC(d):
    return GwcNet(d, use_concat_volume=True)
