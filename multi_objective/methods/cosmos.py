import torch
from torch.distributed.distributed_c10d import is_initialized
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import norm
import numpy as np
import torch.distributed as dist

from collections import OrderedDict

from multi_objective.utils import num_parameters, RunningMean, reference_points, format_list, scale
from .base import BaseMethod

from pymoo.visualization.radviz import Radviz

import matplotlib.pyplot as plt
plt.switch_backend('agg')

np.set_printoptions(precision=2, suppress=True)

class Upsampler(nn.Module):


    def __init__(self, K, child_model, input_dim, upsample_fraction=.75):
        """
        In case of tabular data: append the sampled rays to the data instances (no upsampling)
        In case of image data: use a transposed CNN for the sampled rays.
        """
        super().__init__()
        self.dim = input_dim
        self.fraction = upsample_fraction
        self.K = K

        if len(input_dim) == 1:
            # tabular data
            self.tabular = True
        elif len(input_dim) == 3:
            # image data
            assert input_dim[-2] % 4 == 0 and input_dim[-1] % 4 == 0, 'Spatial image dim must be dividable by 4.'
            self.tabular = False
        else:
            raise ValueError(f"Unknown dataset structure, expected 1 or 3 dimensions, got {input_dim}")

        self.child_model = child_model


    def forward(self, batch):
        x = batch['data']
        
        b = x.shape[0]

        a = batch['alpha']

        if self.tabular:
             result = torch.cat((x, a), dim=1)
        else:
            # use transposed convolution
            # Lessons learned: Does not work with lagrangian multipliers

            a = a.reshape(b, self.K, 1, 1)
            
            target_size = (int(self.dim[-2]*self.fraction), int(self.dim[-1]*self.fraction))
            a = torch.nn.functional.interpolate(a, target_size, mode='nearest')
        
            # Random padding to avoid issues with subsequent batch norm layers.
            result = torch.normal(mean=-1, std=1., size=(b, *self.dim), device=x.device)
            
            # Write x into result tensor
            channels = x.shape[1]
            result[:, 0:channels] = x

            # Write a into the middle of the tensor
            height_start = (result.shape[-2] - a.shape[-2]) // 2
            height_end = (result.shape[-2] - a.shape[-2]) - height_start
            width_start = (result.shape[-1] - a.shape[-1]) // 2
            width_end = (result.shape[-1] - a.shape[-1]) - width_start
            if height_start > 0:
                result[:, channels:, height_start:-height_end, width_start:-width_end] = a
            else:
                result[:, channels:] = a

        return self.child_model(dict(data=result))


class COSMOSMethod(BaseMethod):

    def __init__(self, objectives, model, cfg):
        """
        Instanciate the cosmos solver.

        Args:
            objectives: A list of objectives
            alpha: Dirichlet sampling parameter (list or float)
            lamda: Cosine similarity penalty
            dim: Dimensions of the data
            n_test_rays: The number of test rays used for evaluation.
        """
        super().__init__(objectives, model, cfg)
        self.K = len(objectives)
        self.lambda_lr = cfg.lambda_lr
        self.dampening = cfg.dampening
        self.lambda_clip = cfg.lambda_clipping
        self.loss_mins = torch.tensor(cfg.loss_mins, device=self.device)
        self.loss_maxs = cfg.loss_maxs

        n = cfg.n_train_partitions_cosmos
        self.train_ray_mildening = cfg.train_ray_mildening

        dim = list(cfg.dim)
        dim[0] = dim[0] + self.K

        model.change_input_dim(dim[0])
        self.model = Upsampler(self.K, model, dim, cfg.upsample_ratio).to(self.device)

        self.n_params = num_parameters(self.model)
        print("Number of parameters: {}".format(self.n_params))

        self.sample = reference_points(n, self.K, min=0, max=self.loss_maxs - self.loss_mins.cpu().numpy(), tolerance=self.train_ray_mildening)
        self.lagrangian = [torch.zeros(self.K).cuda() for _ in range(len(self.sample))]

        # for debugging
        self.data = [RunningMean(300) for _ in range(len(self.sample))]
        self.constraints = [RunningMean(1) for _ in range(len(self.sample))]
        
        if len(self.loss_mins) != self.K:
            self.loss_mins = self.loss_mins.repeat(self.K)
        
        if len(self.loss_maxs) != self.K:
            self.loss_maxs = [self.loss_maxs[0] for _ in range(self.K)]
        
    
    def new_epoch(self, e):
        if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
            data = torch.stack(self.lagrangian)
            print(f"lambda mean {data.abs().mean():.04f} std {data.std():.04f} max {data.max()} min {data.min()}")
            print("dimensionality", data.shape)

            if e > 0:
                try:    
                    data = []
                    for i in range(len(self.data)):
                        data.append(np.array(self.data[i].queue).mean(axis=0))
                    
                    const = []
                    for i in range(len(self.constraints)):
                        const.append(list(self.constraints[i].queue)[0])
                    
                    data2 = np.array(const)
                    print(f"constraints mean {np.abs(data2).mean():.04f} std {data2.std():.04f} max {data2.max()} min {data2.min()}")

                    print(np.hstack((np.array(data) + self.loss_mins.cpu().numpy(), data2.reshape(-1, 1))))
                
                    if self.K == 2:
                        # 2 dimensions
                        plt.figure(figsize=(8,8))
                        for i, (x, y) in enumerate(self.sample):
                            plt.arrow(self.loss_mins[0], self.loss_mins[1], x, y)
                            plt.text(self.loss_mins[0] + x, self.loss_mins[1] + y, f"{i}")
                        
                        for i, d in enumerate(data):
                            if np.isscalar(d) and np.isnan(d):
                                continue
                            d += self.loss_mins.cpu().numpy()
                            plt.plot(d[0], d[1], "ro")
                            plt.text(d[0], d[1], f"{i}: {const[i]:.04f}")

                        plt.title(f"epoch {e}")
                        plt.savefig('test')
                        plt.close()
                    
                    else:
                        radviz_plot = Radviz()

                        rays = self.sample
                        
                        valid = []
                        for i, d in enumerate(data):
                            if np.isscalar(d) and np.isnan(d):
                                continue
                            else:
                                valid.append(d)
                        valid = np.array(valid)
                        
                        radviz_plot.add(valid)
                        radviz_plot.save('test')
                        plt.close()
                        
                except:
                    print("Not all rays sampled yet.")

    def step(self, batch):
        b = batch['data'].shape[0]
        i = np.random.choice(list(range(len(self.sample))))
        
        # step 1: sample alphas
        a = self.sample[i]
        a = torch.from_numpy(a.astype(np.float32)).to(self.device)
        batch['alpha'] = a.repeat(b, 1)

        # step 2: calculate loss
        self.model.zero_grad()
        batch.update(self.model(batch))
        task_losses = torch.stack(tuple(self.objectives[t](**batch) for t in self.task_ids)).T - self.loss_mins
        
        # This is the Modified Differential Method of Multipliers
        # Platt & Barr: 1987 (NIPS)
        g_i = task_losses / norm(task_losses) - a / norm(a)

        # thanks to the constraints we could also omit linear scalarization
        loss = a.dot(task_losses.T) + sum(self.lagrangian[i] * g_i) + self.dampening / 2 * sum(g_i ** 2)
        loss.backward()

        const = sum(self.lagrangian[i] * g_i).item()

        # gradient ascent on lagrangian multipliers
        for j, g_j in enumerate(g_i):
            self.lagrangian[i][j] += self.lambda_lr * g_j.item()

            if self.lagrangian[i][j] > self.lambda_clip:
                self.lagrangian[i][j] = self.lambda_clip
            elif self.lagrangian[i][j] < -self.lambda_clip:
                self.lagrangian[i][j] = -self.lambda_clip

        self.data[i].append(task_losses.tolist())
        self.constraints[i].append(const)
        return task_losses.sum().item()



    def eval_step(self, batch, preference_vector):
        self.model.eval()
        with torch.no_grad():
            b = batch['data'].shape[0]
            a = torch.from_numpy(preference_vector).to(self.device).float()
            batch['alpha'] = a.repeat(b, 1)
            return self.model(batch)

    def state_dict(self):
        state = OrderedDict()
        for i, l in enumerate(self.lagrangian):
            state[f'lamdas.{i}'] = l
        return state

    
    def load_state_dict(self, dict):
        self.lagrangian = list(dict.values())


    def preference_at_inference(self):
        return True