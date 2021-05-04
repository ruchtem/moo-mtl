import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import norm
import numpy as np

from multi_objective.utils import num_parameters, RunningMean, reference_points, format_list
from .base import BaseMethod

import matplotlib.pyplot as plt
plt.switch_backend('agg')


class Upsampler(nn.Module):


    def __init__(self, K, child_model, input_dim, upsample_fraction=1.):
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
            self.transposed_cnn = nn.Sequential(
                nn.ConvTranspose2d(K, K, kernel_size=4, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(K, K, kernel_size=6, stride=2, padding=1, bias=False),
            )
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
            a = a.reshape(b, self.K, 1, 1)
            
            # Lessons learned: Does not work with lagrangian multipliers
            # a = self.transposed_cnn(a)

            # plt.hist(a[0].view(-1).cpu().detach())
            # plt.title(f"{batch['alpha'][0]}")
            # plt.savefig('test2')
            # plt.close()

            target_size = (int(self.dim[-2]*self.fraction), int(self.dim[-1]*self.fraction))
            a = torch.nn.functional.interpolate(a, target_size, mode='nearest')
        
            # Random padding to avoid issues with subsequent batch norm layers.
            result = torch.randn(b, *self.dim, device=x.device)
            
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
        self.lambda_lr = cfg.cosmos.lambda_lr
        self.lambda_clip = cfg.cosmos.lambda_clip

        n = cfg.cosmos.n_train_rays

        dim = list(cfg.dim)
        dim[0] = dim[0] + self.K

        self.data = [RunningMean(300) for _ in range(n+1)]

        model.change_input_dim(dim[0])
        self.model = Upsampler(self.K, model, dim).to(self.device)

        self.n_params = num_parameters(self.model)
        print("Number of parameters: {}".format(self.n_params))

        self.steps = 0
        self.sample = reference_points(n, self.K, min=[cfg.cosmos.max_ray_range, cfg.cosmos.max_ray_range])
        self.lagrangian = [torch.zeros(self.K).cuda() for _ in range(len(self.sample))]


    def preference_at_inference(self):
        return True
    

    def new_epoch(self, e):
        if e > 0:
            self.gather = False
            self.loss_mins = torch.stack(self.losses).mean(dim=0) * 0.025

            if (e+1) % 10 == 0:
                self.lagrangian = [torch.zeros(self.K).cuda() for _ in range(len(self.sample))]

            data = []
            for i in range(len(self.data)):
                data.append(np.array(self.data[i].queue).mean(axis=0))
            

            for i, (x, y) in enumerate(self.sample):
                plt.arrow(self.loss_mins[0], self.loss_mins[1], x, y)
                plt.text(self.loss_mins[0] + x, self.loss_mins[1] + y, f"{i}")
            
            for i, d in enumerate(data):
                if np.isscalar(d) and np.isnan(d):
                    continue
                d += self.loss_mins.cpu().numpy()
                plt.plot(d[0], d[1], "ro")
                plt.text(d[0], d[1], f"{i}: {format_list(self.lagrangian[i].tolist(), '.2f')}")

            plt.title(f"epoch {e}")
            plt.savefig('test')
            plt.close()
        else:
            self.gather = True
            self.losses = []


    def step(self, batch):
        b = batch['data'].shape[0]
        if not self.gather:

            # i = np.random.choice(list(range(len(self.sample))), p=[.1, .15, .25, .25, .15, .1])
            i = np.random.choice(list(range(len(self.sample))))
            # i = 0
            
            
            # step 1: sample alphas
            a = self.sample[i]
            a = torch.from_numpy(a.astype(np.float32)).to(self.device)
            batch['alpha'] = a.repeat(b, 1)

            # step 2: calculate loss
            self.model.zero_grad()
            logits = self.model(batch)
            batch.update(logits)
            loss = torch.tensor(0, device=self.device).float()
            
            task_losses = torch.stack(tuple(self.objectives[t](**batch) for t in self.task_ids)).T - self.loss_mins

            # self.loss_mins.append(task_losses.tolist())

            # if self.steps > 10:
            #     loss_mins = np.array(self.loss_mins.queue).min(axis=0)
            #     task_losses -= torch.from_numpy(loss_mins).to(self.device)
            
            g_i = (task_losses / norm(task_losses) - a / norm(a))

            loss = a.dot(task_losses.T) + sum(self.lagrangian[i] * g_i)
            loss.backward()

            test = sum(g_i.abs()).item()

            # gradient ascent on lagrangian multipliers
            for j, g_j in enumerate(g_i):
                self.lagrangian[i][j] += self.lambda_lr * g_j.item()

                if self.lagrangian[i][j] > self.lambda_clip:
                    self.lagrangian[i][j] = self.lambda_clip
                elif self.lagrangian[i][j] < -self.lambda_clip:
                    self.lagrangian[i][j] = -self.lambda_clip

            self.data[i].append(task_losses.tolist())



            self.steps += 1
            return task_losses.sum().item(), test, 0
        else:
            with torch.no_grad():
                a = torch.from_numpy(np.zeros((b, self.K)).astype(np.float32)).to(self.device)
                batch['alpha'] = a

                # step 2: calculate loss
                logits = self.model(batch)
                batch.update(logits)
                
                task_losses = torch.stack(tuple(self.objectives[t](**batch) for t in self.task_ids)).T
                self.losses.append(task_losses)
                return 0



    def eval_step(self, batch, preference_vector):
        self.model.eval()
        with torch.no_grad():
            b = batch['data'].shape[0]
            a = torch.from_numpy(preference_vector).to(self.device).float()
            batch['alpha'] = a.repeat(b, 1)
            return self.model(batch)
