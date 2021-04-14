import torch
import torch.nn as nn
import numpy as np

from utils import num_parameters, calc_gradients, RunningMean
from ..base import BaseMethod


class Upsampler(nn.Module):


    def __init__(self, K, child_model, input_dim, upsample_fraction=1.):
        """
        In case of tabular data: append the sampled rays to the data instances (no upsampling)
        In case of image data: use a transposed CNN for the sampled rays.
        """
        super().__init__()
        self.dim = input_dim
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
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unknown dataset structure, expected 1 or 3 dimensions, got {input_dim}")

        self.child_model = child_model


    def forward(self, batch):
        x = batch['data']
        
        b = x.shape[0]
        a = batch['alpha'].repeat(b, 1)

        if self.tabular:
             result = torch.cat((x, a), dim=1)
        else:
            # use transposed convolution
            a = a.reshape(b, len(batch['alpha']), 1, 1)
            a = self.transposed_cnn(a)

            target_size = (int(self.dim[-2]*self.fraction), int(self.dim[-1]*self.fraction))
            a = torch.nn.functional.interpolate(a, target_size, mode='nearest')
        
            # Random padding to avoid issues with subsequent batch norm layers.
            result = torch.randn(b, *self.dim, device=x.device)
            
            # Write x into result tensor
            channels = x.shape[1]
            result[:, 0:channels] = x

            # Write a into the middle of the tensor
            idx_height = (result.shape[-2] - a.shape[-2]) // 2
            idx_width = (result.shape[-1] - a.shape[-1]) // 2
            if idx_height > 0:
                result[:, channels:, idx_height:-idx_height, idx_width:-idx_width] = a
            else:
                result[:, channels:] = a

        return self.child_model(dict(data=result))


class COSMOSMethod(BaseMethod):

    def __init__(self, objectives, model, alpha, lamda, dim, **kwargs):
        """
        Instanciate the cosmos solver.

        Args:
            objectives: A list of objectives
            alpha: Dirichlet sampling parameter (list or float)
            lamda: Cosine similarity penalty
            dim: Dimensions of the data
            n_test_rays: The number of test rays used for evaluation.
        """
        super().__init__(objectives, model, **kwargs)
        self.K = len(objectives)
        self.alpha = alpha
        self.lamda = lamda

        dim = list(dim)
        dim[0] = dim[0] + self.K

        self.data = RunningMean(100)
        self.lr = kwargs['lr']

        model.change_input_dim(dim[0])
        self.model = Upsampler(self.K, model, dim).to(self.device)

        self.n_params = num_parameters(self.model)
        print("Number of parameters: {}".format(self.n_params))


    def preference_at_inference(self):
        return True


    def step(self, batch):
        # step 1: sample alphas
        if isinstance(self.alpha, list):
            batch['alpha'] = np.random.dirichlet(self.alpha, 1).flatten()
        elif self.alpha > 0:
            batch['alpha'] = np.random.dirichlet([self.alpha for _ in self.task_ids], 1).flatten()
        else:
            raise ValueError(f"Unknown value for alpha: {self.alpha}, expecting list or float.")

        batch['alpha'] = torch.from_numpy(batch['alpha'].astype(np.float32)).to(self.device)

        # replace with gradnorm
        grads, _ = calc_gradients(batch, self.model, self.objectives)

        # step 2: calculate loss
        self.model.zero_grad()
        logits = self.model(batch)
        batch.update(logits)
        loss_total = torch.tensor(0, device=self.device).float()
        task_losses = []
        task_losses_norm = []
        stds = []
        for i, (a, t) in enumerate(zip(batch['alpha'], self.task_ids)):
            task_loss = self.objectives[t](**batch)
            task_loss_norm = task_loss
            if len(self.data) > 2:
                std = self.data.std(axis=0)[i]
                g = grads[t]
                task_loss_norm  = task_loss / sum((g_i**2).sum().sqrt() for g_i in g.values()).sqrt().item()
                # task_loss_norm = task_loss / std    # z normalization (normalize scale)
            loss_total += task_loss_norm * a
            task_losses.append(task_loss.item())
            task_losses_norm.append(task_loss_norm)
        
        if len(self.data) > 2:
            c = sum([a.item() * t for a, t in zip(batch['alpha'], task_losses)]) / sum([a.item() * t.item() for a, t in zip(batch['alpha'], task_losses_norm)])
            # loss_total = loss_total * c
            print(self.data.std(axis=0), c)
        
        # loss_linearization = sum(task_losses).item()
        loss_linearization = sum(task_losses_norm).item()

        cossim = torch.nn.functional.cosine_similarity(torch.stack(task_losses_norm), batch['alpha'], dim=0)
        loss_total -= self.lamda * cossim
        
        loss_total.backward()
        
        self.data.append(task_losses)

        return loss_linearization, cossim.item()


    def eval_step(self, batch, preference_vector):
        self.model.eval()
        with torch.no_grad():
            batch['alpha'] = torch.from_numpy(preference_vector).to(self.device).float()
            return self.model(batch)

