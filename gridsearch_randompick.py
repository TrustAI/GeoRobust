import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import numpy as np

import os
import itertools as it
import argparse
import time

from geo_transf_verifications import GeometricVarification, AffineTransf, obstacle_bound, make_theta, reachability_loss, cw_loss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example-idx', default=506, type=int)
    parser.add_argument('--mnist', action='store_true')
    parser.add_argument('--cifar10', action='store_true')
    parser.add_argument('--imagenet', action='store_true')
    parser.add_argument('--data-dir', default='/mnt/storage0_8/torch_datasets/mnist-data/', type=str)
    parser.add_argument('--model-dir', default='./model', type=str)
    parser.add_argument('--model-name', default='fc_1', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--tss', action='store_true')

    # Transformation
    parser.add_argument('--angle', default=0.0, type=float)
    parser.add_argument('--shift', default=0, type=float)
    parser.add_argument('--scale', default=0.0, type=float)
    parser.add_argument('--obstacle', action='store_true')
    parser.add_argument('--l-inf-bound', default=0.3, type=float)
    parser.add_argument('--topleft-x', default=0, type=int)
    parser.add_argument('--topleft-y', default=0, type=int)
    parser.add_argument('--width', default=0, type=int)
    parser.add_argument('--height', default=0, type=int)

    # baseline setup
    parser.add_argument('--grid-search', action='store_true')
    parser.add_argument('--grid', default=1000, type=int)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--random-pick', action='store_true')
    parser.add_argument('--cw', action='store_true')
    parser.add_argument('--pick-size', default=1000, type=int)

    return parser.parse_args()

def get_partation(data_points, batch_size):
    nb_complete_batches = len(data_points)//batch_size
    batches = []
    for kk in range(nb_complete_batches):
        batches.append(data_points[kk * batch_size: (kk+1) * batch_size])
    if len(data_points) % batch_size != 0:
        batches.append(data_points[(kk+1)*batch_size:])
    return batches

def main():
    args = get_args()

    if args.imagenet:
        from imagenet_utils import prepare_img, load_timm_model
        testset = torchvision.datasets.ImageFolder(args.data_dir)
        raw_img, label = testset[args.example_idx]
        img = prepare_img(raw_img, args.model_name)
        model = load_timm_model(args.model_name, args.device)

    elif args.mnist:
        from mnist_utils import load_mnist, load_model
        _, test_loader = load_mnist(args.data_dir,100,100)
        img, label = test_loader.dataset.__getitem__(args.example_idx)
        if args.tss:
            from tss_utils import load_tss_model
            model = load_tss_model("mnist_43", "mnist", os.path.join(args.model_dir, args.model_name))
            model.eval()
        else:
            model = load_model(args.model_dir, args.model_name, args.device)
    elif args.cifar10:
        from cifar_utils import load_cifar10
        test_loader = load_cifar10(args.data_dir,100)
        img, label = test_loader.dataset.__getitem__(args.example_idx)

        from tss_utils import load_tss_model
        model = load_tss_model("cifar_resnet110", "cifar10", os.path.join(args.model_dir, args.model_name))
        model.eval()

    data_size = tuple(img.shape)
    
    ori_out = model(img.unsqueeze(0).to(args.device))
    ori_conf = cw_loss(ori_out, label).item()
    if torch.argmax(ori_out).item() == label:
        correctness = 1
    else:
        correctness = 0

    transf = []
    bound = []
    location_dist = {}
    if args.angle != 0:
        transf.append('angle')
        bound.append([-np.pi*args.angle, np.pi*args.angle])
    if args.shift != 0:
        transf.append('h_shift'); transf.append('v_shift')
        bound.append([-args.shift, args.shift])
        bound.append([-args.shift, args.shift])
    if args.scale != 0:
        transf.append('scale')
        bound.append([1-args.scale, 1+args.scale])
    assert len(bound) != 0

    task = GeometricVarification(model, img, data_size, label, cw_loss, args.device, transf, **location_dist)
    object_func = task.set_problem()
    optimal_result = None
    minimum = np.inf

    if args.grid_search:
        grid = [None for ii in range(len(bound))]
        for dim,bd in enumerate(bound):
            grid[dim] = np.linspace(bd[0], bd[1], num=args.grid)
        product_obj = '('
        for i in range(len(bound)):
            product_obj += f'grid[{i}],'
        product_obj = product_obj[:-1] + ')'
        comb = np.array(tuple(eval(f'it.product{product_obj}')))
        mini_batches = get_partation(comb, args.batch_size)
        start_time = time.time()
        for bb in mini_batches:
            query_result = object_func(bb)
            tmp_min = np.min(query_result)
            if tmp_min < minimum:
                minimum = tmp_min
                optimal_result = bb[np.argmin(query_result)]
        end_time = time.time()
    elif args.random_pick:
        picked_data = np.random.uniform(0,1,size=(args.pick_size,len(bound)))
        b_value = np.array(bound)
        picked_data = picked_data*(b_value[:,1]-b_value[:,0]) + b_value[:,0]
        mini_batches = get_partation(picked_data, args.batch_size)
        start_time = time.time()
        for bb in mini_batches:
            query_result = object_func(bb)
            tmp_min = np.min(query_result)
            if tmp_min < minimum:
                minimum = tmp_min
                optimal_result = bb[np.argmin(query_result)]
        end_time = time.time()

    opt_theta = make_theta(transf, optimal_result)
    optimal_transf = AffineTransf(opt_theta)
    optimal_img = optimal_transf(img.unsqueeze(0))

    opt_out = model(optimal_img.to(args.device))
    if torch.argmax(opt_out).item() == label:
        post_correctness = 1
    else:
        post_correctness = 0
    print(f'{args.example_idx},{correctness},{ori_conf:.6f},{post_correctness},{minimum:.6f},{(end_time - start_time):.2f}')
    print(f'{optimal_result}')


if __name__ == '__main__':
    main()