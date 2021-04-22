import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from multi_objective.utils import num_parameters, calc_gradients, RunningMean
from .base import BaseMethod


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
            a = self.transposed_cnn(a)

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

        if len(self.alpha) == 1:
            self.alpha = [self.alpha[0] for _ in self.task_ids]

        for k in self.objectives:
            self.objectives[k].reduction = 'none'

        dim = list(cfg.dim)
        dim[0] = dim[0] + self.K

        self.data = RunningMean(2)     # should be updates per epoch
        self.alphas = RunningMean(2)

        model.change_input_dim(dim[0])
        self.model = Upsampler(self.K, model, dim, self.alpha).to(self.device)

        self.bn = {t: torch.nn.BatchNorm1d(1) for t in self.task_ids}

        self.n_params = num_parameters(self.model)
        print("Number of parameters: {}".format(self.n_params))


    def preference_at_inference(self):
        return True
    

    # def new_epoch(self, e):
    #     if e > 2:
            # data = np.array(self.data.queue)
            # self.means.append(data.mean(axis=0))
            # self.stds.append(data.std(axis=0) + 1e-8)

            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(ncols=3)

            # axes[0].hist(data, bins=20, histtype='step')

            # trans = (data-data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
            # axes[1].hist(trans, bins=20, histtype='step')

            # sigm = 1/ (1 + np.exp(-trans))
            # axes[2].hist(sigm, bins=20, histtype='step')
            # plt.savefig(f'hist_{e}')
            # plt.close()


    def step(self, batch):
        b = batch['data'].shape[0]
        # step 1: sample alphas
        a = np.random.dirichlet(self.alpha, size=b)
        a = torch.from_numpy(a.astype(np.float32)).to(self.device)
        batch['alpha'] = a

        # step 2: calculate loss
        self.model.zero_grad()
        logits = self.model(batch)
        batch.update(logits)
        loss = torch.tensor(0, device=self.device).float()
        
        task_losses = torch.stack(tuple(self.objectives[t](**batch) for t in self.task_ids)).T
        
        test = torch.vstack((task_losses.detach(), *list(self.data.queue))) if len(self.data.queue) else task_losses

        min = test.min(dim=0).values
        max = test.max(dim=0).values
        # print(min, max)

        task_losses = (task_losses - min) / (max - min + 1e-8)      # min max norm

        min_a = a.min(dim=0).values.detach()
        max_a = a.max(dim=0).values.detach()

        task_losses = (task_losses * (max_a - min_a)) + min_a       # scale to range of sampled alphas

        # if len(self.data) > 2:
        #     data = np.vstack(tuple(a for a in self.data.queue))
        #     min = torch.from_numpy(data.min(axis=0)).to(self.device)
        #     max = torch.from_numpy(data.max(axis=0)).to(self.device)

        #     task_losses = (task_losses - min) / (max - min + 1e-8)      # min max norm
        #     task_losses = torch.abs(task_losses)
                        
        #     data = np.vstack(tuple(a for a in self.alphas.queue))
        #     min_a = torch.from_numpy(data.min(axis=0)).to(self.device)
        #     max_a = torch.from_numpy(data.max(axis=0)).to(self.device)
            
        #     task_losses = (task_losses * (max_a - min_a)) + min_a       # scale to range of sampled alphas
        
        
        loss = (a * task_losses).sum(dim=1)
        loss_scalar = loss.mean().item()

        cossim = F.cosine_similarity(task_losses, a)
        loss -= self.lamda * cossim

        loss = loss.mean()

        loss.backward()
        
        self.data.append(task_losses.detach())
        self.alphas.append(a.detach())

        return loss_scalar, cossim.mean().item(), 0


    def eval_step(self, batch, preference_vector):
        self.model.eval()
        with torch.no_grad():
            b = batch['data'].shape[0]
            a = torch.from_numpy(preference_vector).to(self.device).float()
            batch['alpha'] = a.repeat(b, 1)
            return self.model(batch)
