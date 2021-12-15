import numpy as np
from scipy import linalg
import torch
from torchvision import models
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms



def fid_torch(model, x_true, x_false, device, mode='bicubic'):
    # resize
    x_true = F.interpolate(x_true, 299, mode=mode)
    x_false = F.interpolate(x_false, 299, mode=mode)
    # get output
    out_true = model(x_true).detach().to('cpu').numpy()
    out_false = model(x_false).detach().to('cpu').numpy()
    # calculate mu, sigma
    mu_true = np.mean(out_true, 0)
    mu_false = np.mean(out_false, 0)
    sigma_true = np.cov(out_true, rowvar=False)
    sigma_false = np.cov(out_false, rowvar=False)
    # calculate fid
    mu_diff = mu_true - mu_false
    cov = linalg.sqrtm(sigma_true.dot(sigma_false)).real
    fid = mu_diff.dot(mu_diff) + np.trace(sigma_true + sigma_false - 2*cov)
    return fid


def fid_td():
    return


def fid_pil():
    return


def fid_cv():
    return