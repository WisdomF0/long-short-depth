import numpy as np
import re
import torchvision.transforms as transforms


def get_transform(normalize=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if normalize:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
        ])

# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines
