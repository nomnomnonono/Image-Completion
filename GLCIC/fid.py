import numpy as np
from scipy import linalg
import torch
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



def fid_pil(model, x_true, x_false, device, mode=Image.BICUBIC):
    resize_true = torch.zeros([1, 3, 299, 299])
    resize_false = torch.zeros([1, 3, 299, 299])
    for i in range(x_true.shape[0]):
      pil_true = transforms.functional.to_pil_image(x_true[i, :, :, :]).resize((299, 299), mode)
      pil_false = transforms.functional.to_pil_image(x_false[i, :, :, :]).resize((299, 299), mode)
      resize_true = torch.cat([resize_true, torch.unsqueeze(transforms.functional.to_tensor(pil_true), 0)], 0)
      resize_false = torch.cat([resize_false, torch.unsqueeze(transforms.functional.to_tensor(pil_false), 0)], 0)
    resize_true = resize_true[1:33, :, :, :].to(device)
    resize_false = resize_false[1:33, :, :, :].to(device)
    # get output
    out_true = model(resize_true).detach().to('cpu').numpy()
    out_false = model(resize_false).detach().to('cpu').numpy()
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



def fid_cv(model, x_true, x_false, device, mode=cv2.INTER_CUBIC):
    x_true = x_true.detach().cpu().numpy().transpose(0, 2, 3, 1)
    x_false = x_false.detach().cpu().numpy().transpose(0, 2, 3, 1)
    resize_true = torch.zeros(32, 3, 299, 299)
    resize_false = torch.zeros(32, 3, 299, 299)
    for i in range(x_true.shape[0]):
        tmp_true = cv2.resize(x_true[i, :, :, :], (299, 299), interpolation=mode)
        tmp_false = cv2.resize(x_false[i, :, :, :], (299, 299), interpolation=mode)
        tmp_true = torch.from_numpy(tmp_true.transpose(2, 0, 1))
        tmp_false = torch.from_numpy(tmp_false.transpose(2, 0, 1))
        resize_true[i, :, :, :] = tmp_true
        resize_false[i, :, :, :] = tmp_false
    resize_true = resize_true.to(device)
    resize_false = resize_false.to(device)
    # get output
    out_true = model(resize_true).detach().to('cpu').numpy()
    out_false = model(resize_false).detach().to('cpu').numpy()
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