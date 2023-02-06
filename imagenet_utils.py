import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os

import timm


tss_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
transform_224 = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(), ])

transform_299 = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(), ])


resnest_list = ['resnet50',
                'resnet152',
                'wide_resnet50_2',
                'wide_resnet101_2'
                'resnet34',
                'resnet101',
                'resnet152d']


inception_list = ['adv_inception_v3',
                  'inception_v3',
                  'inception_v4']

vit_list = ['vit_base_patch16_224', 'vit_large_patch16_224',
            'vit_base_patch32_224', 'vit_small_patch16_224']

mlp_list = ['mixer_b16_224', 'gmlp_s16_224',
            'resmlp_12_224', 'resmlp_big_24_224']

other_list = ['swin_base_patch4_window7_224',
              'swin_large_patch4_window7_224', 
              'beit_base_patch16_224',
              'beit_large_patch16_224',
              'pit_b_224',
              'xcit_large_24_p16_224',
              'xcit_medium_24_p16_224']

_224_models = ['vit_base_patch16_224',
               'mixer_b16_224',
               'gmlp_s16_224',
               'swin_base_patch4_window7_224',
               'pit_b_224',
               'xcit_medium_24_p16_224',
               'beit_base_patch16_224']

_299_models = resnest_list+inception_list

def load_timm_model(model_name, device):
    m = timm.create_model(model_name, pretrained=True)
    m.to(device)
    return m.eval()

def load_model(dir, model_name, device):
    pass

def prepare_img(img, model_name):
    if model_name in _299_models:
        return transform_299(img)
    else:
        return transform_224(img)
