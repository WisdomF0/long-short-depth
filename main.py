import model.model as module
import model.lsdepth as lsdepth
from model.loss_functions import simple_loss
from utils import metrics
from utils.experiment import *

from dataloader.ls_dataset import LongShortDataset
from utils.experiment import adjust_learning_rate

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np

from tensorboardX import SummaryWriter
import gc

data_path = "/home/fengyaohui/data/kitti/"
train_list = "./filenames/mytrain.txt"
test_list = "./filenames/mytest.txt"

logdir = "/home/fengyaohui/src/log/GP/lsdepth"
logger = SummaryWriter(logdir)
summary_freq = 20
save_freq = 1

batch_size = 16
test_batch_size = 8
train_workers = 32
test_workers = 16
lr = 0.01
lrepochs = "10,12,14,16:2"
maxdisp = 192

trainDataset = LongShortDataset(data_path, train_list)
testDataset = LongShortDataset(data_path, test_list)
TrainLoader = DataLoader(trainDataset, batch_size, shuffle=True, num_workers=train_workers, drop_last=True)
TestLoader = DataLoader(testDataset, test_batch_size, shuffle=False, num_workers=test_workers, drop_last=False)

model = module.BruteModel(device="cuda")
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

epochs = 16

for epoch in range(epochs):
    adjust_learning_rate(optimizer, epoch, lr, lrepochs)

    # training
    for batch_idx, data_dict in enumerate(TrainLoader):
        global_step = len(TrainLoader) * epoch + batch_idx
        start_time = time.time()
        do_summary = global_step % summary_freq == 0

        model.train()

        for v in data_dict.values():
            v.cuda()

        optimizer.zero_grad()

        data_dict = model(data_dict)
        mask = (data_dict["ground_truth"] < maxdisp) & (data_dict["ground_truth"] > 0)
        loss = simple_loss.simple_loss(data_dict, mask)

        scalar_outputs = {"loss": loss}
        image_outputs = {"result": data_dict["result"],
                         "result_mono": data_dict["result_mono"],
                         "imgL": data_dict["leftframe"],
                         "imgR": data_dict["rightframe"]}
        with torch.no_grad():
            scalar_outputs["EPE"] = [metrics.EPE_metric(data_dict["result"], data_dict["ground_truth"], mask)]
            scalar_outputs["D1"] = [metrics.D1_metric(data_dict["result"], data_dict["ground_truth"], mask)]
            scalar_outputs["Thres1"] = [metrics.Thres_metric(data_dict["result"], data_dict["ground_truth"], mask, 1.0)]
            scalar_outputs["Thres2"] = [metrics.Thres_metric(data_dict["result"], data_dict["ground_truth"], mask, 2.0)]
            scalar_outputs["Thres3"] = [metrics.Thres_metric(data_dict["result"], data_dict["ground_truth"], mask, 3.0)]

        loss.backward()
        optimizer.step()

        loss = tensor2float(loss)
        scalar_outputs = tensor2float(scalar_outputs)

        if do_summary:
            save_scalars(logger, 'train', scalar_outputs, global_step)
            save_images(logger, 'train', image_outputs, global_step)
        del scalar_outputs, image_outputs
        print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch, epochs,
                                                                                       batch_idx,
                                                                                       len(TrainLoader), loss,
                                                                                       time.time() - start_time))
    
    if (epoch + 1) % save_freq == 0:
        checkpoint_data = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(logdir, epoch))
    gc.collect()

    # testing
    avg_test_scalars = AverageMeterDict()
    for batch_idx, data_dict in enumerate(TestLoader):
        global_step = len(TestLoader) * epoch + batch_idx
        start_time = time.time()
        do_summary = global_step % summary_freq == 0

        model.eval()
        for v in data_dict.values():
            v.cuda()
        
        data_dict = model(data_dict)
        mask = (data_dict["ground_truth"] < maxdisp) & (data_dict["ground_truth"] > 0)
        loss = simple_loss.simple_loss(data_dict, mask)

        scalar_outputs = {"loss": loss}
        image_outputs = {"result": data_dict["result"],
                         "result_mono": data_dict["result_mono"],
                         "imgL": data_dict["leftframe"],
                         "imgR": data_dict["rightframe"]}
        
        scalar_outputs["EPE"] = [metrics.EPE_metric(data_dict["result"], data_dict["ground_truth"], mask)]
        scalar_outputs["D1"] = [metrics.D1_metric(data_dict["result"], data_dict["ground_truth"], mask)]
        scalar_outputs["Thres1"] = [metrics.Thres_metric(data_dict["result"], data_dict["ground_truth"], mask, 1.0)]
        scalar_outputs["Thres2"] = [metrics.Thres_metric(data_dict["result"], data_dict["ground_truth"], mask, 2.0)]
        scalar_outputs["Thres3"] = [metrics.Thres_metric(data_dict["result"], data_dict["ground_truth"], mask, 3.0)]

        loss = tensor2float(loss)
        scalar_outputs = tensor2float(scalar_outputs)

        if do_summary:
            save_scalars(logger, 'test', scalar_outputs, global_step)
            save_images(logger, 'test', image_outputs, global_step)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch, epochs,
                                                                                 batch_idx,
                                                                                 len(TestLoader), loss,
                                                                                 time.time() - start_time))

    avg_test_scalars = avg_test_scalars.mean()
    save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainLoader) * (epoch + 1))
    print("avg_test_scalars", avg_test_scalars)
    gc.collect()


