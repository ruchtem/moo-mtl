import torch
import torch.nn as nn
import numpy as np

from utils import num_parameters, RunningMinMaxNormalizer
from ..base import BaseMethod


class Upsampler(nn.Module):


    def __init__(self, K, child_model, input_dim):
        """
        In case of tabular data: append the sampled rays to the data instances (no upsampling)
        In case of image data: use a transposed CNN for the sampled rays.
        """
        super().__init__()

        if len(input_dim) == 1:
            # tabular data
            self.tabular = True
        elif len(input_dim) == 3:
            # image data
            self.tabular = False
            self.transposed_cnn = nn.Sequential(
                nn.ConvTranspose2d(K, K, kernel_size=4, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(K, K, kernel_size=6, stride=2, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Upsample(input_dim[-2:])
            )
        else:
            raise ValueError(f"Unknown dataset structure, expected 1 or 3 dimensions, got {input_dim}")

        self.child_model = child_model


    def forward(self, batch):
        x = batch['data']
        
        b = x.shape[0]
        a = batch['alpha'].repeat(b, 1)
        
        if not self.tabular:
            # use transposed convolution
            a = a.reshape(b, len(batch['alpha']), 1, 1)
            a = self.transposed_cnn(a)
        
        x = torch.cat((x, a), dim=1)
        return self.child_model(dict(data=x))


class COSMOSMethod(BaseMethod):

    def __init__(self, objectives, model, alpha, lamda, dim, normalize_rays, **kwargs):
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
        self.normalize_rays = normalize_rays
        if normalize_rays:
            self.loss_normalizer = RunningMinMaxNormalizer()
            self.s = 0

        dim = list(dim)
        dim[0] = dim[0] + self.K

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

        # normalize rays:
        if self.normalize_rays and self.s > 50:
           batch['alpha'] = self.loss_normalizer.normalize(batch['alpha'])

        batch['alpha'] = torch.from_numpy(batch['alpha'].astype(np.float32)).to(self.device)

        
        # step 2: calculate loss
        self.model.zero_grad()
        logits = self.model(batch)
        batch.update(logits)
        loss_total = None
        task_losses = []
        for a, t in zip(batch['alpha'], self.task_ids):
            task_loss = self.objectives[t](**batch)
            loss_total = a * task_loss if not loss_total else loss_total + a * task_loss
            task_losses.append(task_loss)
        
        cossim = torch.nn.functional.cosine_similarity(torch.stack(task_losses), batch['alpha'], dim=0)
        loss_total -= self.lamda * cossim
            
        loss_total.backward()

        if self.normalize_rays:
            self.loss_normalizer.update(task_losses)
            self.s += 1
        return loss_total.item(), cossim.item()


    def eval_step(self, batch, preference_vector):
        self.model.eval()
        if self.normalize_rays:
                preference_vector = self.loss_normalizer.normalize(preference_vector)
        with torch.no_grad():
            batch['alpha'] = torch.from_numpy(preference_vector).to(self.device).float()
            return self.model(batch)

