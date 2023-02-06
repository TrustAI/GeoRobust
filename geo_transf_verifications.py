import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineTransf(nn.Module):

    def __init__(self, theta):
        super(AffineTransf, self).__init__()
        self.theta = theta.view(1,2,3)
        
    def forward(self, img):
        batch_size, channel, height, width = img.size()

        grid = F.affine_grid(self.theta, torch.Size((
            batch_size, channel, height, width)))
        transfored_img = F.grid_sample(img, grid)
        return transfored_img

def obstacle_bound(X, tl_x, tl_y, w, h, linf_b):
    up_bound = (1 - X.squeeze()[tl_y:tl_y+h,tl_x:tl_x+w]).view(-1).numpy()
    low_bound = (0 - X.squeeze()[tl_y:tl_y+h,tl_x:tl_x+w]).view(-1).numpy()
    l_inf_bound = linf_b - 1e-5
    up_bound[up_bound > l_inf_bound] = l_inf_bound
    low_bound[low_bound < -l_inf_bound] = -l_inf_bound
    bounds = list(zip(low_bound,up_bound))
    return bounds

def _3_channel_obstacle_bound(X, tl_x, tl_y, w, h, linf_b):
    print(X.shape)
    up_bound = (1 - X[:,tl_y:tl_y+h,tl_x:tl_x+w]).view(-1).numpy()
    low_bound = (0 - X[:,tl_y:tl_y+h,tl_x:tl_x+w]).view(-1).numpy()
    print(up_bound.shape)
    l_inf_bound = linf_b - 1e-5
    up_bound[up_bound > l_inf_bound] = l_inf_bound
    low_bound[low_bound < -l_inf_bound] = -l_inf_bound
    bounds = list(zip(low_bound,up_bound))
    return bounds

def reachability_loss(outputs, y):
    return outputs[:,y]

def cw_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    
    loss_value = x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
    return loss_value

def make_theta(transf, arr):

    if 'angle' in transf:
        if type(arr) == torch.Tensor:
            aff_alpha = torch.cos(arr[transf.index('angle')])
            aff_beta = torch.sin(arr[transf.index('angle')])
        else:
            aff_alpha = np.cos(arr[transf.index('angle')])
            aff_beta = np.sin(arr[transf.index('angle')])
    else:
        aff_alpha = 1.
        aff_beta = 1.
    if 'v_shift' in transf:
        x_shift = arr[transf.index('h_shift')]
        y_shift = arr[transf.index('v_shift')]
    else:
        x_shift = 0.
        y_shift = 0.
    if 'scale' in transf:
        aff_alpha = aff_alpha * arr[-1]
        aff_beta = aff_beta * arr[-1]

    theta = torch.tensor([aff_alpha,
                           -aff_beta,
                           x_shift,
                           aff_beta,
                           aff_alpha,
                           y_shift], dtype=torch.float32)
    return theta



class GeometricVarification():
    def __init__(self, 
                 model,
                 image,
                 data_size,
                 true_label, 
                 verify_loss,
                 device,
                 transfms,
                 **kwargs):

        self.model = model
        self.ori_example = image
        self.size = data_size
        self.loss = verify_loss
        self.y = true_label
        self.device = device

        for k,v in kwargs.items():
            exec(f'self.{k.lower()} = {v}')

        self.mark = ''
        if transfms == 'obstacle':
            self.mark = '1'
            if self.size[0] > 1: self.mark = '10'
            
        else:
            for k in ['angle', 'h_shift', 'v_shift', 'scale']:
                if k in transfms:
                    self.mark += '1'
                else:
                    self.mark += '0'

    def set_problem(self):
        if self.mark == '1': return self.obstacle_verification
        elif self.mark == '10': return self._3_channel_obstacle_verification
        elif self.mark == '1000': return self.angle_verification
        elif self.mark == '0110': return self.shift_verification
        elif self.mark == '0001': return self.scale_verification
        elif self.mark == '1110': return self.angle_shift_verification
        elif self.mark == '0111': return self.shift_scale_verification
        elif self.mark == '1001': return self.angle_scale_verification
        elif self.mark == '1111': return self.angle_shift_scale_verification

    def obstacle_verification(self, in_arrs):
        tmp_tensor = torch.ones((len(in_arrs),self.size[0],self.size[1],self.size[2]))
        for idx,in_arr in enumerate(in_arrs):
            obstacle_area = torch.zeros_like(self.ori_example.squeeze())
            obstacle_area[self.tl_y:self.tl_y+self.height,self.tl_x:self.tl_x+self.width] = torch.tensor(in_arr).view(self.height,self.width)
            tmp_tensor[idx] = obstacle_area
        with torch.no_grad():
            out = self.model(tmp_tensor.to(self.device))
            loss = self.loss(out,self.y)
        return loss.detach().cpu().numpy()

    def _3_channel_obstacle_verification(self, in_arrs):
        tmp_tensor = torch.ones((len(in_arrs),self.size[0],self.size[1],self.size[2]))
        for idx,in_arr in enumerate(in_arrs):
            obstacle_area = torch.zeros_like(self.ori_example.squeeze())
            obstacle_area[:,self.tl_y:self.tl_y+self.height,self.tl_x:self.tl_x+self.width] = torch.tensor(in_arr).view(self.size[0], self.height,self.width)
            tmp_tensor[idx] = obstacle_area
        with torch.no_grad():
            out = self.model(tmp_tensor.to(self.device))
            loss = self.loss(out,self.y)
        return loss.detach().cpu().numpy()

    def angle_verification(self, in_arrs):
        tmp_tensor = torch.ones((len(in_arrs),self.size[0],self.size[1],self.size[2]))
        for idx,in_arr in enumerate(in_arrs):
            aff_alpha = np.cos(in_arr).item()
            aff_beta = np.sin(in_arr).item()
            cur_theta = torch.tensor([aff_alpha,
                                      -aff_beta,
                                      0,
                                      aff_beta,
                                      aff_alpha,
                                      0], dtype=torch.float32)
            cur_aff = AffineTransf(cur_theta)
            tmp_tensor[idx] = cur_aff(self.ori_example.unsqueeze(0))
        with torch.no_grad():
            out = self.model(tmp_tensor.to(self.device))
            loss = self.loss(out,self.y)
        return loss.detach().cpu().numpy()

    def shift_verification(self, in_arrs):
        tmp_tensor = torch.ones((len(in_arrs),self.size[0],self.size[1],self.size[2]))
        for idx,in_arr in enumerate(in_arrs):
            x_shift, y_shift = in_arr[0], in_arr[1]
            cur_theta = torch.tensor([1,0,x_shift,
                                      0,1,y_shift], dtype=torch.float32)
            cur_aff = AffineTransf(cur_theta)
            tmp_tensor[idx] = cur_aff(self.ori_example.unsqueeze(0))
        with torch.no_grad():
            out = self.model(tmp_tensor.to(self.device))
            loss = self.loss(out,self.y)
        return loss.detach().cpu().numpy()

    def scale_verification(self, in_arrs):
        tmp_tensor = torch.ones((len(in_arrs),self.size[0],self.size[1],self.size[2]))
        for idx,scaler in enumerate(in_arrs):
            cur_theta = torch.tensor([scaler.item(),0,0,
                                      0,scaler.item(),0], dtype=torch.float32)
            cur_aff = AffineTransf(cur_theta)
            tmp_tensor[idx] = cur_aff(self.ori_example.unsqueeze(0))
        with torch.no_grad():
            out = self.model(tmp_tensor.to(self.device))
            loss = self.loss(out,self.y)
        return loss.detach().cpu().numpy()

    def angle_shift_verification(self, in_arrs):
        tmp_tensor = torch.ones((len(in_arrs),self.size[0],self.size[1],self.size[2]))
        for idx,in_arr in enumerate(in_arrs):
            angle, x_shift, y_shift = in_arr[0], in_arr[1], in_arr[2]
            aff_alpha = np.cos(angle).item()
            aff_beta = np.sin(angle).item()
            cur_theta = torch.tensor([aff_alpha,
                                      -aff_beta,
                                      x_shift,
                                      aff_beta,
                                      aff_alpha,
                                      y_shift], dtype=torch.float32)
            cur_aff = AffineTransf(cur_theta)
            tmp_tensor[idx] = cur_aff(self.ori_example.unsqueeze(0))
        with torch.no_grad():
            out = self.model(tmp_tensor.to(self.device))
            loss = self.loss(out,self.y)
        return loss.detach().cpu().numpy()

    def shift_scale_verification(self, in_arrs):
        tmp_tensor = torch.ones((len(in_arrs),self.size[0],self.size[1],self.size[2]))
        for idx,in_arr in enumerate(in_arrs):
            x_shift, y_shift, scaler = in_arr[0],in_arr[1],in_arr[2]
            cur_theta = torch.tensor([scaler,0,x_shift,
                                      0,scaler,y_shift], dtype=torch.float32)
            cur_aff = AffineTransf(cur_theta)
            tmp_tensor[idx] = cur_aff(self.ori_example.unsqueeze(0))
        with torch.no_grad():
            out = self.model(tmp_tensor.to(self.device))
            loss = self.loss(out,self.y)
        return loss.detach().cpu().numpy()

    def angle_scale_verification(self, in_arrs):
        tmp_tensor = torch.ones((len(in_arrs),self.size[0],self.size[1],self.size[2]))
        for idx,in_arr in enumerate(in_arrs):
            angle, scaler = in_arr[0], in_arr[1]
            aff_alpha = scaler*np.cos(angle).item()
            aff_beta = scaler*np.sin(angle).item()
            cur_theta = torch.tensor([aff_alpha,
                                      -aff_beta,
                                      0,
                                      aff_beta,
                                      aff_alpha,
                                      0], dtype=torch.float32)
            cur_aff = AffineTransf(cur_theta)
            tmp_tensor[idx] = cur_aff(self.ori_example.unsqueeze(0))
        with torch.no_grad():
            out = self.model(tmp_tensor.to(self.device))
            loss = self.loss(out,self.y)
        return loss.detach().cpu().numpy()

    def angle_shift_scale_verification(self, in_arrs):
        tmp_tensor = torch.ones((len(in_arrs),self.size[0],self.size[1],self.size[2]))
        for idx,in_arr in enumerate(in_arrs):
            angle, x_shift, y_shift, scaler= in_arr[0], in_arr[1], in_arr[2], in_arr[3]
            aff_alpha = scaler * np.cos(angle)
            aff_beta = scaler * np.sin(angle)
            cur_theta = torch.tensor([aff_alpha,
                                    -aff_beta,
                                    x_shift,
                                    aff_beta,
                                    aff_alpha,
                                    y_shift], dtype=torch.float32)
            cur_aff = AffineTransf(cur_theta)
            tmp_tensor[idx] = cur_aff(self.ori_example.unsqueeze(0))
        with torch.no_grad():
            out = self.model(tmp_tensor.to(self.device))
            loss = self.loss(out,self.y)
        return loss.detach().cpu().numpy()

