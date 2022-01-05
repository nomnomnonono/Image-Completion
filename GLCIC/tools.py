import random
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2



def generate_mask(shape, hole_size, hole_area):
    mask = torch.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape
    
    for i in range(bsize):
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


def generate_multiple_mask(shape, hole_size, hole_area, n_holes):
    mask = torch.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape
    limit = mask_w // 6
    for i in range(bsize):
        for _ in range(n_holes):
            # define hole width
            if isinstance(hole_size[0], tuple) and len(hole_size[0]) == 2:
                hole_w = random.randint(hole_size[0][0] / np.sqrt(n_holes), hole_size[0][1] / np.sqrt(n_holes))
            else:
                hole_w = hole_size[0]
            # define hole height
            if isinstance(hole_size[1], tuple) and len(hole_size[1]) == 2:
                hole_h = random.randint(hole_size[1][0] / np.sqrt(n_holes), hole_size[1][1] / np.sqrt(n_holes))
            else:
                hole_h = hole_size[1]
            # define hole-arae
            area_xmin, area_ymin = hole_area
            offset_x = random.randint(area_xmin, area_xmin + mask_w - hole_w)
            if offset_x < limit:
              offset_x = limit
            if offset_x > mask_w - limit - hole_w:
              offset_x = mask_w - limit - hole_w
            offset_y = random.randint(area_ymin, area_ymin + mask_h - hole_h)
            if offset_y < limit:
              offset_y = limit
            if offset_y > mask_h - limit - hole_h:
              offset_y = mask_h - limit - hole_h
            # generate loss-area
            mask[i, :, offset_y: offset_y + hole_h, offset_x: offset_x + hole_w] = 1.0
    return mask


def generate_circle_mask(shape, radius_range, hole_area, n_holes):
    # radius_range=(30, 40)
    mask = np.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape
    limit = mask_w // 6
    for i in range(bsize):
        for _ in range(n_holes):
            # define hole width
            if isinstance(radius_range, tuple) and len(radius_range) == 2:
                #hole_w = int(random.randint(hole_size[0][0], hole_size[0][1]) / np.sqrt(n_holes))
                radius = random.randint(radius_range[0] / np.sqrt(n_holes), radius_range[1] / np.sqrt(n_holes))
            else:
                radius = radius_range
            # define hole-arae
            area_xmin, area_ymin = hole_area
            offset_x = random.randint(area_xmin, area_xmin + mask_w - radius)
            if offset_x < limit:
              offset_x = limit
            if offset_x > mask_w - limit - radius:
              offset_x = mask_w - limit - radius
            offset_y = random.randint(area_ymin, area_ymin + mask_h - radius)
            if offset_y < limit:
              offset_y = limit
            if offset_y > mask_h - limit - radius:
              offset_y = mask_h - limit - radius
            # generate loss-area
            cv2.circle(mask[i, 0, :, :], (offset_x, offset_y), radius, 1, thickness=-1)
    mask = torch.from_numpy(mask)
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