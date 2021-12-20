import random
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2



def generate_mask(shape, hole_size, hole_area, max_holes=1):
    """
    inputs:
        - shape
        - hole_size(sequence, required):
            ((hole_min_w, hole_max_w), (hole_min_h, hole_max_h))
        - hole_area(tuple, optional):
            This argument constraints the area where holes are generated.
            tupple = (upper-left-x, upper-left-y)
        - max_holes(int, optional):
            This argument specifies how many holes are generated.
            The number of holes is randomly chosen from [1, max_holes].
            The default value is 1.
    returns:
        A mask tensor of shape [N, C, H, W] with holes.
        All the pixel values within holes are filled with 1.0,
        while the other pixel values are zeros.
    """
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
    xmin, ymin = area[0]
    w, h = area[1]
    return x[:, :, ymin: ymin + h, xmin: xmin + w]


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