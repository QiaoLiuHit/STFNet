"""
Description: spectrum loss for GAN
"""
import torch
import pdb
import torch.nn.functional as F
import cv2
import sys
import numpy as np
from torch.autograd import Variable
sys.path.append("..")


def calc_fft(image):
    '''image is tensor, N*C*H*W'''
    fft = torch.fft.fft2(image, dim=(-2, -1))
    # fft_mag = torch.log(1 + torch.sqrt(fft[..., 0] ** 2 + fft[..., 1] ** 2 + 1e-8))
    fft_mag = torch.log(1 + torch.sqrt(fft.real ** 2 + fft.imag ** 2 + 1e-8))
    return fft_mag


def fft_L1_loss(fake_image, real_image):
    criterion_L1 = torch.nn.L1Loss()

    fake_image_gray = fake_image[:, 0] * 0.299 + fake_image[:, 1] * 0.587 + fake_image[:, 2] * 0.114
    real_image_gray = real_image[:, 0] * 0.299 + real_image[:, 1] * 0.587 + real_image[:, 2] * 0.114

    fake_fft = calc_fft(fake_image_gray)
    real_fft = calc_fft(real_image_gray)
    loss = criterion_L1(fake_fft, real_fft)
    return loss


def fft_L1_loss_mask(fake_image, real_image, mask):
    criterion_L1 = torch.nn.L1Loss()
    # fake_image_gray = fake_image[:, 0] * 0.299 + fake_image[:, 1] * 0.587 + fake_image[:, 2] * 0.114
    # real_image_gray = real_image[:, 0] * 0.299 + real_image[:, 1] * 0.587 + real_image[:, 2] * 0.114
    fake_fft = calc_fft(fake_image)
    real_fft = calc_fft(real_image)
    loss = criterion_L1(fake_fft * mask, real_fft * mask)
    return loss


def fft_L1_loss_color(fake_image, real_image):
    criterion_L1 = torch.nn.L1Loss()

    fake_fft = calc_fft(fake_image)
    real_fft = calc_fft(real_image)
    loss = criterion_L1(fake_fft, real_fft)
    return loss


def decide_circle(N=4, L=256, r=96, size=256):
    x = torch.ones((N, L, L))
    for i in range(L):
        for j in range(L):
            if (i - L / 2 + 0.5) ** 2 + (j - L / 2 + 0.5) ** 2 < r ** 2:
                x[:, i, j] = 0
    return x, torch.ones((N, L, L)) - x




def get_gaussian_blur(x, k, stride=1, padding=0):
    res = []
    x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
    for xx in x.split(1, 1):
        res.append(F.conv2d(xx, k, stride=stride, padding=0))
    return torch.cat(res, 1)

def get_low_freq(im, gauss_kernel):
    padding = (gauss_kernel.shape[-1] - 1) // 2
    low_freq = get_gaussian_blur(im, gauss_kernel, padding=padding)
    return low_freq


def gaussian_blur(x, k, stride=1, padding=0):
    res = []
    x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
    for xx in x.split(1, 1):
        res.append(F.conv2d(xx, k, stride=stride, padding=0))
    return torch.cat(res, 1)

def get_gaussian_kernel(size=3):
    kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
    return kernel


def cal_low_freq(im, gauss_kernel, index=None):
    padding = (gauss_kernel.shape[-1] - 1) // 2
    low_freq = gaussian_blur(im, gauss_kernel, padding=padding)
    # im_gray = im[:, 0, ...] * 0.299 + im[:, 1, ...] * 0.587 + im[:, 2, ...] * 0.114
    # im_gray = im_gray.unsqueeze_(dim=1).repeat(1, 3, 1, 1)
    # low_gray = gaussian_blur(im_gray, gauss_kernel, padding=padding)
    # return torch.cat((low_freq, im_gray - low_gray),1)
    return low_freq

# im is a cuda img
def gradient(im):
    laplace_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype='float32')
    laplace_kernel = laplace_kernel.reshape((1, 1, 3, 3))
    weight = Variable(torch.from_numpy(laplace_kernel)).cuda()
    output = F.conv2d(im, weight, padding=1)
    return output