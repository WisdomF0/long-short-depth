import json, os
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy import sparse
from torch.utils.data import Dataset

from utils import map_fn
import torchvision.transforms.functional as F

from dataloader.data_io import get_transform, read_all_lines

class LongShortDataset(Dataset):
    def __init__(self, data_path, list_filename, image_size=(192, 640), use_color=True):
        self.data_path = data_path
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.image_size = image_size
        self.use_color = use_color
        self.processed = get_transform(normalize=True)
    
    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images
        
    def load_image(self, filename) -> torch.Tensor:
        image =  Image.open(filename).convert('RGB')

        if self.image_size:
            image = image.resize((self.image_size[1], self.image_size[0]), resample=Image.BILINEAR)
        
        image_tensor = self.processed(image)
        # image_tensor = torch.tensor(np.array(image).astype(np.float32))
        # image_tensor = image_tensor / 255 - .5
        # if not self.use_color:
        #     image_tensor = torch.stack((image_tensor, image_tensor, image_tensor))
        # else:
        #     image_tensor = image_tensor.permute(2, 0, 1)
        del image
        return image_tensor
    
    def load_disp(self, filename):
        data = Image.open(filename)

        if self.image_size:
            data = data.resize((self.image_size[1], self.image_size[0]), resample=Image.BILINEAR)

        data = np.array(data, dtype=np.float32) / 256.
        return torch.tensor(data)
    
    def __len__(self):
        return len(self.left_filenames)
    
    def __getitem__(self, index):
        leftframe = self.load_image(os.path.join(self.data_path, self.left_filenames[index]))
        rightframe = self.load_image(os.path.join(self.data_path, self.right_filenames[index]))
        depth = self.load_disp(os.path.join(self.data_path, self.disp_filenames[index]))

        data_dict = {
            "leftframe": leftframe,
            "rightframe": rightframe,
            "ground_truth": depth
        }

        return data_dict
