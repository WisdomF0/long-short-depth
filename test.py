import model.model as module
import model.lsdepth as lsdepth

import torch
from torchsummary import summary
import numpy as np

def testMono():
    model = module.MonoDepthModule()
    feature_extractor = module.ResnetEncoder(num_layers=18, pretrained=False)

    data_dict = {
        "leftframe": torch.randn(1, 3, 192, 640),
    }

    data_dict["left_features"] = feature_extractor(data_dict["leftframe"] + .5)

    output = model(data_dict)

    inv_depth_min_max = (0.33, 0.0025)
    data_dict["predicted_inverse_depths_mono"] = [(1-pred) * inv_depth_min_max[1] + pred * inv_depth_min_max[0]
                                                for pred in data_dict["predicted_inverse_depths_mono"]]
    mono_depth_pred = torch.clamp(1.0 / data_dict["predicted_inverse_depths_mono"][0], min=1e-3, max=80.0).detach()

    print(mono_depth_pred.shape)

    def pseudocost_from_mono(monodepth, depth_hypothesis):
        abs_depth_diff = torch.abs(monodepth - depth_hypothesis)
        # find the closest depth bin that the monodepth correlate with
        min_diff_index = torch.argmin(abs_depth_diff, dim=1, keepdim=True)
        pseudo_cost = depth_hypothesis.new_zeros(depth_hypothesis.shape)
        ones = depth_hypothesis.new_ones(depth_hypothesis.shape)
        
        pseudo_cost.scatter_(dim = 1, index = min_diff_index, src = ones)
        
        return pseudo_cost

    cv_depth_steps = 32
    data_dict["inv_depth_min"] = data_dict["leftframe"].new_tensor([inv_depth_min_max[0]])
    data_dict["inv_depth_max"] = data_dict["leftframe"].new_tensor([inv_depth_min_max[1]])
    data_dict["cv_depth_steps"] = data_dict["leftframe"].new_tensor([cv_depth_steps], dtype=torch.int32)
    depths = (1 / torch.linspace(data_dict["inv_depth_max"][0].item(), 
                                data_dict["inv_depth_min"][0].item(), 
                                data_dict["cv_depth_steps"][0].item()))
    data_dict["cv_bin_steps"] = depths

    b, c, h, w = data_dict["leftframe"].shape

    pseudo_mono_cost = pseudocost_from_mono(mono_depth_pred, 
                                        depth_hypothesis = data_dict["cv_bin_steps"].view(1, -1, 1, 1).expand(b, -1, h, w).detach()).detach()

    print(pseudo_mono_cost.shape)

def testGWCNet():
    d = 192
    model = module.GwcNet(d, True)
    model = torch.nn.DataParallel(model)
    model.cuda()

    ckpt_path = "/home/fengyaohui/src/log/GP/gwc_linear/checkpoint_000023.ckpt"
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict['model'])
    model.eval()

    for key in state_dict['model']:
        print(key)

    data_dict = {
        "leftframe": torch.randn(2, 3, 256, 256),
        "rightframe": torch.randn(2, 3, 256, 256),
    }

    for k, v in data_dict.items():
        v.cuda()

    out = model(data_dict["leftframe"], data_dict["rightframe"])
    print(out[0].shape)

    out = out[0].unsqueeze(1)
    print(out.shape)

def testFusion():
    h, w = 192, 640
    mono_cost = torch.randn(2, 32, h, w)
    stereo_cost = torch.randn(2, 32, h, w)

    feature_extractor = module.ResnetEncoder(num_layers=18, pretrained=False)

    image = torch.randn(2, 3, h, w)

    features =  feature_extractor(image + .5)

    model = module.BruteFusion(feature_channels=feature_extractor.num_ch_enc)

    out = model(mono_cost, stereo_cost, features)

    print(len(out))

def testGWC():
    model = module.GWCStereoModule(128, use_concat_volume=True)

    ckpt_path = "/home/fengyaohui/src/log/GP/gwc_linear/checkpoint_000023.ckpt"
    state_dict = torch.load(ckpt_path)
    params_dict = {}
    for key, param in state_dict['model'].items():
        if "feature_extraction" in key:
            params_dict[key] = param

    # print(params_dict.keys())
    # model = torch.nn.DataParallel(model)
    # model.cuda()

    model.load_state_dict(params_dict, strict=False)

    data_dict = {
        "leftframe": torch.randn(1, 3, 192, 640),
        "rightframe": torch.randn(1, 3, 192, 640),
    }

    # for v in data_dict.values():
    #     v.cuda()

    out = model(data_dict)
    B, C, D, H, W = out.shape
    print(out.shape)

    amodel = module.GWCDModule(C, 1)
    # amodel.cuda()
    out = amodel(out)
    print(out.shape)

    out = out.squeeze(1)
    print(out.shape)

def testModule():
    data_dict = {
        "leftframe": torch.randn(1, 3, 192, 640),
        "rightframe": torch.randn(1, 3, 192, 640),
    }

    model = module.BruteModel(backbone_pretrain=False)
    # model.cuda()

    # for v in data_dict.values():
    #     v.cuda()

    data_dict = model(data_dict)
    print("result:", data_dict["result"].shape)
    print("result mono:", data_dict["result_mono"].shape)

testModule()

# for k, v in output.items():
#     if not isinstance(v, torch.Tensor):
#         print(len(v))
#         v = v[0]
#     print(k, v.shape)
