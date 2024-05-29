from __future__ import absolute_import, division, print_function

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F, Conv2d, LeakyReLU, Upsample, Sigmoid, ConvTranspose2d, Conv3d, ConvTranspose3d


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))

def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))

def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)

def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class Conv3x3(nn.Module):
    """一个卷积核为三的卷积层，添加一个pad=1的padding层来使形状不变，通过use_refl参数来决定是否使用反射填充
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out
    
class Backprojection(nn.Module):
    """将深度图逆投影到3D点云中
    """
    def __init__(self, batch_size, height, width):
        super(Backprojection, self).__init__()

        self.N, self.H, self.W = batch_size, height, width

        yy, xx = torch.meshgrid([torch.arange(0., float(self.H)), torch.arange(0., float(self.W))])
        yy = yy.contiguous().view(-1)
        xx = xx.contiguous().view(-1)
        self.ones = nn.Parameter(torch.ones(self.N, 1, self.H * self.W), requires_grad=False)
        self.coord = torch.unsqueeze(torch.stack([xx, yy], 0), 0).repeat(self.N, 1, 1)
        self.coord = nn.Parameter(torch.cat([self.coord, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K) :
        cam_p_norm = torch.matmul(inv_K[:, :3, :3], self.coord[:depth.shape[0], :, :])
        cam_p_euc = depth.view(depth.shape[0], 1, -1) * cam_p_norm
        cam_p_h = torch.cat([cam_p_euc, self.ones[:depth.shape[0], :, :]], 1)

        return cam_p_h
    
def point_projection(points3D, batch_size, height, width, K, T):
    """点云投影到2D平面
    """
    N, H, W = batch_size, height, width
    cam_coord = torch.matmul(torch.matmul(K, T)[:, :3, :], points3D)
    img_coord = cam_coord[:, :2, :] / (cam_coord[:, 2:3, :] + 1e-7)
    img_coord[:, 0, :] /= W - 1
    img_coord[:, 1, :] /= H - 1
    img_coord = (img_coord - 0.5) * 2
    img_coord = img_coord.view(N, 2, H, W).permute(0, 2, 3, 1)
    return img_coord

def upsample(x, scale=2):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale, mode="nearest")


class GaussianAverage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.window = torch.Tensor([
            [0.0947, 0.1183, 0.0947],
            [0.1183, 0.1478, 0.1183],
            [0.0947, 0.1183, 0.0947]])

    def forward(self, x):
        kernel = self.window.to(x.device).to(x.dtype).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, kernel, padding=0, groups=x.shape[1])

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self, pad_reflection=True, gaussian_average=False, comp_mode=False):
        super(SSIM, self).__init__()
        self.comp_mode = comp_mode

        if not gaussian_average:
            self.mu_x_pool   = nn.AvgPool2d(3, 1)
            self.mu_y_pool   = nn.AvgPool2d(3, 1)
            self.sig_x_pool  = nn.AvgPool2d(3, 1)
            self.sig_y_pool  = nn.AvgPool2d(3, 1)
            self.sig_xy_pool = nn.AvgPool2d(3, 1)
        else:
            self.mu_x_pool = GaussianAverage()
            self.mu_y_pool = GaussianAverage()
            self.sig_x_pool = GaussianAverage()
            self.sig_y_pool = GaussianAverage()
            self.sig_xy_pool = GaussianAverage()

        if pad_reflection:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.pad(x)
        y = self.pad(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_x_y = mu_x * mu_y

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x_sq
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y_sq
        sigma_xy = self.sig_xy_pool(x * y) - mu_x_y

        SSIM_n = (2 * mu_x_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x + sigma_y + self.C2)

        if not self.comp_mode:
            return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        else:
            return torch.clamp((1 - SSIM_n / SSIM_d), 0, 1) / 2


def ssim(x, y, pad_reflection=True, gaussian_average=False, comp_mode=False):
    ssim_ = SSIM(pad_reflection, gaussian_average, comp_mode)
    return ssim_(x, y)


class PadSameConv2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=1):
        """
        Imitates padding_mode="same" from tensorflow.
        :param kernel_size: Kernelsize of the convolution, int or tuple/list
        :param stride: Stride of the convolution, int or tuple/list
        """
        super().__init__()
        if isinstance(kernel_size, (tuple, list)):
            self.kernel_size_y = kernel_size[0]
            self.kernel_size_x = kernel_size[1]
        else:
            self.kernel_size_y = kernel_size
            self.kernel_size_x = kernel_size
        if isinstance(stride, (tuple, list)):
            self.stride_y = stride[0]
            self.stride_x = stride[1]
        else:
            self.stride_y = stride
            self.stride_x = stride

    def forward(self, x: torch.Tensor):
        _, _, height, width = x.shape

        # For the convolution we want to achieve a output size of (n_h, n_w) = (math.ceil(h / s_y), math.ceil(w / s_y)).
        # Therefore we need to apply n_h convolution kernels with stride s_y. We will have n_h - 1 offsets of size s_y.
        # Additionally, we need to add the size of our kernel. This is the height we require to get n_h. We need to pad
        # the read difference between this and the old height. We will pad math.floor(pad_y / 2) on the left and
        # math-ceil(pad_y / 2) on the right. Same  for pad_x respectively.
        padding_y = (self.stride_y * (math.ceil(height / self.stride_y) - 1) + self.kernel_size_y - height) / 2
        padding_x = (self.stride_x * (math.ceil(width / self.stride_x) - 1) + self.kernel_size_x - width) / 2
        padding = [math.floor(padding_x), math.ceil(padding_x), math.floor(padding_y), math.ceil(padding_y)]
        return F.pad(input=x, pad=padding)
    

class PadSameConv2dTransposed(torch.nn.Module):
    def __init__(self, stride):
        """
        Imitates padding_mode="same" from tensorflow.
        :param stride: Stride of the convolution_transposed, int or tuple/list
        """
        super().__init__()
        if isinstance(stride, (tuple, list)):
            self.stride_y = stride[0]
            self.stride_x = stride[1]
        else:
            self.stride_y = stride
            self.stride_x = stride

    def forward(self, x: torch.Tensor, orig_shape: torch.Tensor):
        target_shape = x.new_tensor(list(orig_shape))
        target_shape[-2] *= self.stride_y
        target_shape[-1] *= self.stride_x
        oversize = target_shape[-2:] - x.new_tensor(x.shape)[-2:]
        if oversize[0] > 0 and oversize[1] > 0:
            x = F.pad(x, [math.floor(oversize[1] / 2), math.ceil(oversize[1] / 2), math.floor(oversize[0] / 2),
                          math.ceil(oversize[0] / 2)])
        elif oversize[0] > 0 >= oversize[1]:
            x = F.pad(x, [0, 0, math.floor(oversize[0] / 2), math.ceil(oversize[0] / 2)])
            x = x[:, :, :, math.floor(-oversize[1] / 2):-math.ceil(-oversize[1] / 2)]
        elif oversize[0] <= 0 < oversize[1]:
            x = F.pad(x, [math.floor(oversize[1] / 2), math.ceil(oversize[1] / 2)])
            x = x[:, :, math.floor(-oversize[0] / 2):-math.ceil(-oversize[0] / 2), :]
        else:
            x = x[:, :, math.floor(-oversize[0] / 2):-math.ceil(-oversize[0] / 2),
                math.floor(-oversize[1] / 2):-math.ceil(-oversize[1] / 2)]
        return x
    

class PadSameConv3dTransposed(torch.nn.Module):
    def __init__(self, stride):
        super().__init__()
        if isinstance(stride, (tuple, list)):
            self.stride_y = stride[0]
            self.stride_x = stride[1]
        else:
            self.stride_y = stride
            self.stride_x = stride

    def forward(self, x: torch.Tensor, orig_shape: torch.Tensor):
        target_shape = x.new_tensor(list(orig_shape))
        target_shape[-2] *= self.stride_y
        target_shape[-1] *= self.stride_x
        oversize = target_shape[-2:] - x.new_tensor(x.shape)[-2:]
        if oversize[0] > 0 and oversize[1] > 0:
            x = F.pad(x, [math.floor(oversize[1] / 2), math.ceil(oversize[1] / 2), math.floor(oversize[0] / 2),
                          math.ceil(oversize[0] / 2)])
        elif oversize[0] > 0 >= oversize[1]:
            x = F.pad(x, [0, 0, math.floor(oversize[0] / 2), math.ceil(oversize[0] / 2)])
            x = x[:, :, :, math.floor(-oversize[1] / 2):-math.ceil(-oversize[1] / 2)]
        elif oversize[0] <= 0 < oversize[1]:
            x = F.pad(x, [math.floor(oversize[1] / 2), math.ceil(oversize[1] / 2)])
            x = x[:, :, math.floor(-oversize[0] / 2):-math.ceil(-oversize[0] / 2), :]
        else:
            x = x[:, :, math.floor(-oversize[0] / 2):-math.ceil(-oversize[0] / 2),
                math.floor(-oversize[1] / 2):-math.ceil(-oversize[1] / 2)]
        return x

class Refine3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, leaky_relu_neg_slope=0.1):
        super().__init__()
        self.conv2d_t = ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2)
        self.pad = PadSameConv2dTransposed(stride=2)
        self.leaky_relu = LeakyReLU(negative_slope=leaky_relu_neg_slope)

    

class ConvReLU2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky_relu_neg_slope=0.1):
        """
        Performs two convolutions and a leaky relu. The first operation only convolves in y direction, the second one
        only in x direction.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Kernel size for the convolutions, first in y direction, then in x direction
        :param stride: Stride for the convolutions, first in y direction, then in x direction
        """
        super().__init__()
        self.pad_0 = PadSameConv2d(kernel_size=(kernel_size, 1), stride=(stride, 1))
        self.conv_y = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                             stride=(stride, 1))
        self.leaky_relu = LeakyReLU(negative_slope=leaky_relu_neg_slope)
        self.pad_1 = PadSameConv2d(kernel_size=(1, kernel_size), stride=(1, stride))
        self.conv_x = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                             stride=(1, stride))

    def forward(self, x: torch.Tensor):
        t = self.pad_0(x)
        t = self.conv_y(t)
        t = self.leaky_relu(t)
        t = self.pad_1(t)
        t = self.conv_x(t)
        return self.leaky_relu(t)
    
class ConvReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky_relu_neg_slope=0.1):
        """
        Performs two convolutions and a leaky relu. The first operation only convolves in y direction, the second one
        only in x direction.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Kernel size for the convolutions, first in y direction, then in x direction
        :param stride: Stride for the convolutions, first in y direction, then in x direction
        """
        super().__init__()
        self.pad = PadSameConv2d(kernel_size=kernel_size, stride=stride)
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.leaky_relu = LeakyReLU(negative_slope=leaky_relu_neg_slope)

    def forward(self, x: torch.Tensor):
        t = self.pad(x)
        t = self.conv(t)
        return self.leaky_relu(t)
    

class Upconv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Performs two convolutions and a leaky relu. The first operation only convolves in y direction, the second one
        only in x direction.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Kernel size for the convolutions, first in y direction, then in x direction
        :param stride: Stride for the convolutions, first in y direction, then in x direction
        """
        super().__init__()
        self.upsample = Upsample(scale_factor=2)
        self.pad = PadSameConv2d(kernel_size=2)
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=1)

    def forward(self, x: torch.Tensor):
        t = self.upsample(x)
        t = self.pad(t)
        return self.conv(t)
    
class ConvSig(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        """
        Performs two convolutions and a leaky relu. The first operation only convolves in y direction, the second one
        only in x direction.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Kernel size for the convolutions, first in y direction, then in x direction
        :param stride: Stride for the convolutions, first in y direction, then in x direction
        """
        super().__init__()
        self.pad = PadSameConv2d(kernel_size=kernel_size, stride=stride)
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.sig = Sigmoid()

    def forward(self, x: torch.Tensor):
        t = self.pad(x)
        t = self.conv(t)
        return self.sig(t)
    

class Refine(torch.nn.Module):
    def __init__(self, in_channels, out_channels, leaky_relu_neg_slope=0.1):
        """
        Performs a transposed conv2d with padding that imitates tensorflow same behaviour. The transposed conv2d has
        parameters kernel_size=4 and stride=2.
        :param in_channels: Channels that go into the conv2d_transposed
        :param out_channels: Channels that come out of the conv2d_transposed
        """
        super().__init__()
        self.conv2d_t = ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2)
        self.pad = PadSameConv2dTransposed(stride=2)
        self.leaky_relu = LeakyReLU(negative_slope=leaky_relu_neg_slope)

    def forward(self, x: torch.Tensor, features_direct=None):
        orig_shape=x.shape
        x = self.conv2d_t(x)
        x = self.leaky_relu(x)
        x = self.pad(x, orig_shape)
        if features_direct is not None:
            x = torch.cat([x, features_direct], dim=1)
        return x
    

class SimpleUp(torch.nn.Module):
    def __init__(self, scale=2, mode='bilinear'):
        super(SimpleUp, self).__init__()
        self.scale = scale
        self.mode = mode

        self.up = Upsample(scale_factor=scale, mode=mode)

    def forward(self, x):
        _, _, OH, OW = x.shape

        startH = OH // 4
        startW = OW // 4
        endH = startH + OH // 2
        endW = startW + OW // 2

        x = x[:, :, startH:endH, startW:endW]
        out = self.up(x)

        _, _, H, W = out.shape

        if H != OH or W != OW:
            out = out[:, :, :OH, :OW]
        
        return out
    

class PadSameConv3d(torch.nn.Module):
    def __init__(self, kernel_size, stride=1):
        """
        Imitates padding_mode="same" from tensorflow.
        :param kernel_size: Kernelsize of the convolution, int or tuple/list
        :param stride: Stride of the convolution, int or tuple/list
        """
        super().__init__()
        if isinstance(kernel_size, (tuple, list)):
            self.kernel_size_z = kernel_size[0]
            self.kernel_size_y = kernel_size[1]
            self.kernel_size_x = kernel_size[2]
        else:
            self.kernel_size_z = kernel_size
            self.kernel_size_y = kernel_size
            self.kernel_size_x = kernel_size
        if isinstance(stride, (tuple, list)):
            self.stride_z = stride[0]
            self.stride_y = stride[1]
            self.stride_x = stride[2]
        else:
            self.stride_z = stride
            self.stride_y = stride
            self.stride_x = stride

    def forward(self, x: torch.Tensor):
        _, _, depth, height, width = x.shape

        # For the convolution we want to achieve a output size of (n_h, n_w) = (math.ceil(h / s_y), math.ceil(w / s_y)).
        # Therefore we need to apply n_h convolution kernels with stride s_y. We will have n_h - 1 offsets of size s_y.
        # Additionally, we need to add the size of our kernel. This is the height we require to get n_h. We need to pad
        # the read difference between this and the old height. We will pad math.floor(pad_y / 2) on the left and
        # math-ceil(pad_y / 2) on the right. Same  for pad_x respectively.
        padding_z = (self.stride_z * (math.ceil(depth / self.stride_z) - 1) + self.kernel_size_z - depth) / 2
        padding_y = (self.stride_y * (math.ceil(height / self.stride_y) - 1) + self.kernel_size_y - height) / 2
        padding_x = (self.stride_x * (math.ceil(width / self.stride_x) - 1) + self.kernel_size_x - width) / 2
        padding = [math.floor(padding_x), math.ceil(padding_x), 
                   math.floor(padding_y), math.ceil(padding_y),
                   math.floor(padding_z), math.ceil(padding_z)]
        return F.pad(input=x, pad=padding)


class PadSameConv3dTransposed(torch.nn.Module):
    def __init__(self, stride):
        """
        Imitates padding_mode="same" from tensorflow.
        :param stride: Stride of the convolution_transposed, int or tuple/list
        """
        super().__init__()
        if isinstance(stride, (tuple, list)):
            self.stride_z = stride[0]
            self.stride_y = stride[1]
            self.stride_x = stride[2]
        else:
            self.stride_z = stride
            self.stride_y = stride
            self.stride_x = stride

    def forward(self, x: torch.Tensor, orig_shape: torch.Tensor):
        target_shape = x.new_tensor(list(orig_shape))
        target_shape[-3] *= self.stride_z
        target_shape[-2] *= self.stride_y
        target_shape[-1] *= self.stride_x
        oversize = target_shape[-3:] - x.new_tensor(x.shape)[-3:]

        padding = [0, 0, 0, 0, 0, 0]
        if oversize[2] > 0:
            padding[4], padding[5] = math.floor(oversize[0] / 2), math.ceil(oversize[0] / 2)
        if oversize[1] > 0:
            padding[2], padding[3] = math.floor(oversize[1] / 2), math.ceil(oversize[1] / 2)
        if oversize[0] > 0:
            padding[0], padding[1] = math.floor(oversize[2] / 2), math.ceil(oversize[2] / 2)

        x = F.pad(x, padding)

        if oversize[2] <= 0:
            x = x[:, :, :, :, math.floor(-oversize[2] / 2):-math.ceil(-oversize[2] / 2)]
        if oversize[1] <= 0:
            x = x[:, :, :, math.floor(-oversize[1] / 2):-math.ceil(-oversize[1] / 2), :]
        if oversize[0] <= 0:
            x = x[:, :, math.floor(-oversize[0] / 2):-math.ceil(-oversize[0] / 2), :, :]

        return x
    

class Conv3DReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky_relu_neg_slope=0.1):
        super().__init__()
        self.pad = PadSameConv3d(kernel_size=kernel_size, stride=stride)
        self.conv = Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.leaky_relu = LeakyReLU(negative_slope=leaky_relu_neg_slope)

    def forward(self, x: torch.Tensor):
        t = self.pad(x)
        t = self.conv(t)
        return self.leaky_relu(t)
    
class Conv3DSig(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.pad = PadSameConv3d(kernel_size=kernel_size, stride=stride)
        self.conv = Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.sig = Sigmoid()

    def forward(self, x: torch.Tensor):
        t = self.pad(x)
        t = self.conv(t)
        return self.sig(t)
    
class Refine3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, leaky_relu_neg_slope=0.1):
        super().__init__()
        self.conv3d_t = ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2)
        self.pad = PadSameConv3dTransposed(stride=2)
        self.leaky_relu = LeakyReLU(negative_slope=leaky_relu_neg_slope)

    def forward(self, x: torch.Tensor, features_direct=None):
        orig_shape=x.shape
        x = self.conv3d_t(x)
        x = self.leaky_relu(x)
        x = self.pad(x, orig_shape)
        if features_direct is not None:
            x = torch.cat([x, features_direct], dim=1)
        return x
        