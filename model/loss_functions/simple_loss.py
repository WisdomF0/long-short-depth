import torch.nn.functional as F


def simple_loss(data_dict, mask):
    return 0.7 * F.smooth_l1_loss(data_dict["result_mono"][mask], data_dict["ground_truth"][mask], size_average=True) + 1.0 * F.smooth_l1_loss(data_dict["result"][mask], data_dict["ground_truth"][mask], size_average=True)

def model_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)
