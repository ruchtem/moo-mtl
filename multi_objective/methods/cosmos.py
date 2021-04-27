import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import norm
import numpy as np

from multi_objective.utils import num_parameters, calc_gradients, RunningMean
from .base import BaseMethod

import matplotlib.pyplot as plt
plt.switch_backend('agg')


class Upsampler(nn.Module):


    def __init__(self, K, child_model, input_dim, alpha, upsample_fraction=1.):
        """
        In case of tabular data: append the sampled rays to the data instances (no upsampling)
        In case of image data: use a transposed CNN for the sampled rays.
        """
        super().__init__()
        self.dim = input_dim
        self.alpha = alpha
        self.fraction = upsample_fraction

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
            a = a.reshape(b, len(self.alpha), 1, 1)
            
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
        self.alpha = cfg.cosmos.alpha
        self.lamda = cfg.cosmos.lamda
        self.normalize = cfg.cosmos.normalize

        if len(self.alpha) == 1:
            self.alpha = [self.alpha[0] for _ in self.task_ids]

        for k in self.objectives:
            self.objectives[k].reduction = 'none'

        dim = list(cfg.dim)
        dim[0] = dim[0] + self.K

        self.data = [RunningMean(300) for _ in range(5)]    # arbitrary
        self.alphas = RunningMean(2)

        model.change_input_dim(dim[0])
        self.model = Upsampler(self.K, model, dim, self.alpha).to(self.device)

        self.bn = {t: torch.nn.BatchNorm1d(1) for t in self.task_ids}

        self.n_params = num_parameters(self.model)
        print("Number of parameters: {}".format(self.n_params))

        self.steps = 0
        self.sample = np.random.dirichlet(self.alpha, size=5)
        self.lagrangian = [torch.nn.parameter.Parameter(torch.zeros(2), requires_grad=False).cuda() for _ in range(5)]


    def preference_at_inference(self):
        return True
    

    def new_epoch(self, e):
        if e > 0:

            
            data0 = torch.vstack(tuple(self.data[0].queue)).cpu().numpy().mean(axis=0)
            # data1 = torch.vstack(tuple(self.data[1].queue)).cpu().numpy().mean(axis=0)
            data2 = torch.vstack(tuple(self.data[2].queue)).cpu().numpy().mean(axis=0)
            data3 = torch.vstack(tuple(self.data[3].queue)).cpu().numpy().mean(axis=0)
            data4 = torch.vstack(tuple(self.data[4].queue)).cpu().numpy().mean(axis=0)

            for i, (x, y) in enumerate(self.sample):
                plt.arrow(0, 0, x, y)
                plt.text(x, y, f"{i}")
            
            plt.plot(data0[0], data0[1], "ro")
            plt.text(data0[0], data0[1], "0")
            # plt.plot(data1[0], data1[1], "ro")
            # plt.text(data1[0], data1[1], "1")
            plt.plot(data2[0], data2[1], "ro")
            plt.text(data2[0], data2[1], "2")
            plt.plot(data3[0], data3[1], "ro")
            plt.text(data3[0], data3[1], "3")
            plt.plot(data4[0], data4[1], "ro")
            plt.text(data4[0], data4[1], "4")

            plt.title(f"epoch {e}")
            plt.savefig('test')
            plt.close()


    def step(self, batch):

        i = np.random.choice([0, 2, 3, 4])
        # i = 0
        

        b = batch['data'].shape[0]
        # step 1: sample alphas
        # a = np.random.dirichlet(self.alpha, size=b)
        a = self.sample[i]
        a = torch.from_numpy(a.astype(np.float32)).to(self.device)
        batch['alpha'] = a.repeat(b, 1)  

        # step 2: calculate loss
        self.model.zero_grad()
        logits = self.model(batch)
        batch.update(logits)
        loss = torch.tensor(0, device=self.device).float()
        
        task_losses = torch.stack(tuple(self.objectives[t](**batch) for t in self.task_ids)).T
        
        if self.normalize:
            loss_history = torch.vstack((task_losses.detach(), *list(self.data.queue))) if len(self.data.queue) else task_losses.detach()
            min = loss_history.min(dim=0).values
            max = loss_history.max(dim=0).values

            task_losses_norm = (task_losses - min) / (max - min + 1e-8)      # min max norm

            alpha_history = torch.vstack((a.detach(), *list(self.alphas.queue))) if len(self.alphas.queue) else a.detach()
            min_a = alpha_history.min(dim=0).values
            max_a = alpha_history.max(dim=0).values

            task_losses_norm = (task_losses_norm * (max_a - min_a)) + min_a       # scale to range of sampled alphas
        else:
            task_losses_norm = task_losses
        

        task_losses = task_losses.mean(dim=0)
        g_i = (task_losses / norm(task_losses) - a / norm(a))
        # g_i = F.cosine_similarity(task_losses, a, dim=-1)       # cossim = 1 for angle zero

        loss = a.dot(task_losses.T) + sum(self.lagrangian[i] * g_i)
        loss.backward()

        test = sum(self.lagrangian[i] * g_i).item()

        # gradient ascent on lagrangian multipliers
        lag_lr = 1e-1
        self.lagrangian[i][0] += lag_lr * g_i[0].item()
        self.lagrangian[i][1] += lag_lr * g_i[1].item()
        print(self.lagrangian)
        # print(g_i)



        # task_losses_norm1 = task_losses / torch.linalg.norm(task_losses, dim=1).unsqueeze(1)
        # a_norm = a / torch.linalg.norm(a, dim=1).unsqueeze(1)

        # # loss = torch.linalg.norm(task_losses_norm1 - a_norm, dim=1)
        # loss = torch.pow(task_losses_norm1 - a_norm, 2).sum(dim=1)


        loss_scalar = task_losses.mean().item()

        # task_losses_norm1 = task_losses / torch.linalg.norm(task_losses, dim=1).unsqueeze(1)
        # a_norm = a / torch.linalg.norm(a, dim=1).unsqueeze(1)

        # dist = torch.linalg.norm(task_losses_norm1 - a_norm, dim=1)



        # cossim = F.cosine_similarity(task_losses, a)
        # angular_dist = torch.arccos(cossim) / np.pi

        # loss += self.lamda * dist


        # loss = loss.mean()

        # loss.backward()

        self.data[i].append(task_losses.detach())
        self.alphas.append(a.detach())



        self.steps += 1
        return loss_scalar, test, 0


    def eval_step(self, batch, preference_vector):
        self.model.eval()
        with torch.no_grad():
            b = batch['data'].shape[0]
            a = torch.from_numpy(preference_vector).to(self.device).float()
            batch['alpha'] = a.repeat(b, 1)
            return self.model(batch)
