import random
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2



def generate_mask(shape, hole_size, hole_area, max_holes=1):
    mask = torch.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape
    
    for i in range(bsize):
        # define number of holes
        n_holes = random.choice(list(range(1, max_holes+1)))
        for _ in range(n_holes):
            # define hole width
            if isinstance(hole_size[0], tuple) and len(hole_size[0]) == 2:
                hole_w = random.randint(hole_size[0][0], hole_size[0][1])
            else:
                hole_w = hole_size[0]

            # define hole height
            if isinstance(hole_size[1], tuple) and len(hole_size[1]) == 2:
                hole_h = random.randint(hole_size[1][0], hole_size[1][1])
            else:
                hole_h = hole_size[1]
            area_xmin, area_ymin = hole_area
            offset_x = random.randint(area_xmin, area_xmin + mask_w - hole_w)
            offset_y = random.randint(area_ymin, area_ymin + mask_h - hole_h)
            mask[i, :, offset_y: offset_y + hole_h, offset_x: offset_x + hole_w] = 1.0
    return mask


def define_hole_area(hole_size, mask_size):
    mask_w, mask_h = mask_size
    hole_w, hole_h = hole_size
    offset_x = random.randint(0, mask_w - hole_w)
    offset_y = random.randint(0, mask_h - hole_h)
    return ((offset_x, offset_y), (hole_w, hole_h))


def crop(x, area):
    offset_x, offset_y = area[0]
    w, h = area[1]
    return x[:, :, offset_y: offset_y + h, offset_x: offset_x + w]


def sample_random_batch(dataset, batch_size=32):
    num_samples = len(dataset)
    batch = []
    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        x = torch.unsqueeze(dataset[index], dim=0)  # [1, C, H, W]
        batch.append(x)  # [n, 1, C, H, W]
    return torch.cat(batch, dim=0)  # [n, C, H, W]



def get_completion_image(input, output, mask):
    return input - input*mask + output*mask