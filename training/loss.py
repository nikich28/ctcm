import math

import torch
import torch.nn as nn
from torch_utils import persistence
from torch_utils import distributed as dist
import numpy as np

#----------------------------------------------------------------------------
# Loss function proposed in the blog "Consistency Models Made Easy"

@persistence.persistent_class
class ECMLoss:
    def __init__(self, P_mean=-1.1, P_std=2.0, sigma_data=0.5, q=2, c=0.0, k=8.0, b=1.0, cut=4.0, adj='sigmoid'):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
        if adj == 'const':
            self.t_to_r = self.t_to_r_const
        elif adj == 'sigmoid':
            self.t_to_r = self.t_to_r_sigmoid
        else:
            raise ValueError(f'Unknow schedule type {adj}!')

        self.q = q
        self.stage = 0
        self.ratio = 0.
        
        self.k = k
        self.b = b

        self.c = c
        dist.print0(f'P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}')

    def update_schedule(self, stage):
        self.stage = stage
        self.ratio = 1 - 1 / self.q ** (stage+1)

    def t_to_r_const(self, t):
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay
        r = t * ratio
        return torch.clamp(r, min=0)

    def t_to_r_sigmoid(self, t):
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        # t ~ p(t) and r ~ p(r|t, iters) (Mapping fn)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        r = self.t_to_r(t)

        # Augmentation if needed
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
        # Shared noise direction
        eps   = torch.randn_like(y)
        eps_t = eps * t
        eps_r = eps * r
        
        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        D_yt = net(y + eps_t, t, labels, augment_labels=augment_labels)
        
        if r.max() > 0:
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                D_yr = net(y + eps_r, r, labels, augment_labels=augment_labels)
            
            mask = r > 0
            D_yr = torch.nan_to_num(D_yr)
            D_yr = mask * D_yr + (~mask) * y
        else:
            D_yr = y

        # L2 Loss
        loss = (D_yt - D_yr) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        
        # Producing Adaptive Weighting (p=0.5) through Huber Loss
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)
        
        # Weighting fn
        return loss / (t - r).flatten()


@persistence.persistent_class
class CTCMLoss:
    def __init__(self, P_mean=-1.0, P_std=1.4, sigma_data=0.5, c=0.1, ct=True, pretrained=None, H=10000):
        self.P_mean = P_mean
        self.P_mean_end = P_mean + 0.8
        self.P_std = P_std
        self.sigma_data = sigma_data

        self.c = c
        self.H = H
        
        self.ct = ct
        if not ct:
            self.pretrained = pretrained
        
        self.iters = 0
        
    def update_schedule(self, stage):
        pass

    def normalize(self, x, dim=None, eps=1e-4):
        if dim is None:
            dim = list(range(1, x.ndim))
        norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
        norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
        return x / (norm.to(x.dtype) + self.c)

    def __call__(self, net, images, labels=None, augment_pipe=None, train_ratio=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)

        if (train_ratio is not None) and (self.P_mean != self.P_mean_end):
            p_mean = self.P_mean + train_ratio * (self.P_mean_end - self.P_mean)
        else:
            p_mean = self.P_mean
        tau = (rnd_normal * self.P_std + p_mean).exp()
        t = torch.arctan(tau / self.sigma_data)
        
        self.iters += 1

        # Augmentation if needed
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
        
        z = torch.randn_like(y) * self.sigma_data
        
        
        x_t = torch.cos(t) * y + torch.sin(t) * z
        
        if self.ct:
            dxt = torch.cos(t) * z - torch.sin(t) * y
        else:
            with torch.no_grad():
                dxt = self.sigma_data * self.pretrained(x_t / self.sigma_data, t.flatten(), labels)
            
        
        r = min(1, self.iters / self.H)
        
        def f(a, b):
            return net(a, b.flatten())
        
        output, F_theta_grad, weight_ = torch.func.jvp(f, (x_t / self.sigma_data, t), 
                                            (torch.cos(t)*torch.sin(t)*dxt/self.sigma_data, torch.cos(t)*torch.sin(t)),
                                            has_aux=True)
        weight_ = weight_.view(-1, 1, 1, 1)
        F_theta = output.detach()
        F_theta_grad = F_theta_grad.detach()
        
        g = -torch.cos(t)* torch.cos(t) * (self.sigma_data * F_theta - dxt) - r * torch.cos(t) * torch.sin(t) * x_t - r * self.sigma_data * F_theta_grad
        
        g = self.normalize(g)
    
        loss = torch.square(output - F_theta - g)
        # loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1) # or mean?
        
        loss = loss * weight_.exp() / tau - weight_
        # loss = loss * weight_.exp() / 1.0 - weight_
        
        return loss.flatten()

    @torch.inference_mode()
    def loss_timesteps(self, net, images, num_steps, labels=None, augment_pipe=None, train_ratio=None):
        losses = []
        loss_now = []
        loss_t1 = []
        loss_nowt1 = []
        loss_alt = []
        loss_alt_raw = []

        # Augmentation if needed
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        z = torch.randn_like(y) * self.sigma_data

        timesteps = torch.linspace(0, 1.56454, num_steps, device=images.device)

        for t in timesteps:
            t = t.item() * torch.ones([images.shape[0], 1, 1, 1], device=images.device)
            tau = torch.tan(t) * self.sigma_data

            x_t = torch.cos(t) * y + torch.sin(t) * z
        
            if self.ct:
                dxt = torch.cos(t) * z - torch.sin(t) * y
            else:
                with torch.no_grad():
                    dxt = self.sigma_data * self.pretrained(x_t / self.sigma_data, t.flatten(), labels)
                
            
            r = min(1, self.iters / self.H)
            
            def f(a, b):
                return net(a, b.flatten())
            
            output, F_theta_grad, weight_ = torch.func.jvp(f, (x_t / self.sigma_data, t), 
                                                (torch.cos(t)*torch.sin(t)*dxt/self.sigma_data, torch.cos(t)*torch.sin(t)),
                                                has_aux=True)
            weight_ = weight_.view(-1, 1, 1, 1)
            F_theta = output.detach()
            F_theta_grad = F_theta_grad.detach()
            
            g = -torch.cos(t)* torch.cos(t) * (self.sigma_data * F_theta - dxt) - r * torch.cos(t) * torch.sin(t) * x_t - r * self.sigma_data * F_theta_grad
            g = self.normalize(g)
            
            loss = torch.square(output - F_theta - g)
            # loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1) # or mean?
            
            # loss = loss * weight_.exp() / tau - weight_
            # loss = loss / tau

            # Weighting fn
            losses.append((loss * weight_.exp() / tau - weight_).mean())
            loss_now.append((loss / tau).mean())
            loss_t1.append((loss * weight_.exp() - weight_).mean())
            loss_nowt1.append((loss).mean())

            f_s = torch.cos(t) * x_t - torch.sin(t) * self.sigma_data * output
            dfdt = -torch.sin(t) * x_t - torch.cos(t) * self.sigma_data * F_theta
            dfdx = torch.cos(t) * dxt - self.sigma_data * F_theta_grad / torch.cos(t)
            mc = (x_t - y) / t
            loss_a = 2 * (f_s * (dfdt + dfdx * mc))

            loss_alt.append((loss_a / t).mean())
            loss_alt_raw.append((2 * loss_a).mean())

        return (torch.stack(losses).numpy(force=True), 
                torch.stack(loss_now).numpy(force=True),
                torch.stack(loss_t1).numpy(force=True),
                torch.stack(loss_nowt1).numpy(force=True),
                torch.stack(loss_alt).numpy(force=True),
                torch.stack(loss_alt_raw).numpy(force=True),
                timesteps.numpy(force=True))

@persistence.persistent_class
class CTDiffLoss:
    def __init__(self, P_mean=-1.1, P_std=1.6, sigma_data=0.5, c=0.1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.c = c

    def __call__(self, net, images, augment_pipe=None, labels=None, iters=0):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        tau = (rnd_normal * self.P_std + self.P_mean).exp()
        t = torch.arctan(tau / self.sigma_data)

        # Augmentation if needed
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
        z = torch.randn_like(y) * self.sigma_data
        x_t = torch.cos(t) * y + torch.sin(t) * z
        
        pred, weight_ = net(x_t / self.sigma_data, t.flatten())
        weight_ = weight_.view(-1, 1, 1, 1)
        pred = self.sigma_data * pred
        
        v_t = torch.cos(t) * z - torch.sin(t) * y
        
        loss = (pred - v_t) ** 2
        # loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        
        loss = (loss / self.sigma_data) * weight_.exp() / tau - weight_
        # loss = (1 / torch.exp(weight_)) * (loss / (self.sigma_data**2)) + weight_
        return loss.mean()
