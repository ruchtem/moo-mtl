import torch
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

from multi_objective.min_norm_solvers import MinNormSolver

from multi_objective.utils import calc_gradients, reference_points, get_lr_scheduler, num_parameters
from .base import BaseMethod


def get_d_paretomtl_init(grads, losses, preference_vectors, pref_idx):
    """ 
    calculate the gradient direction for ParetoMTL initialization 

    Args:
        grads: flattened gradients for each task
        losses: values of the losses for each task
        preference_vectors: all preference vectors u
        pref_idx: which index of u we are currently using
    
    Returns:
        flag: is a feasible initial solution found?
        weight: 
    """
    
    flag = False
    nobj = losses.shape
   
    # check active constraints, Equation 7
    current_pref = preference_vectors[pref_idx]         # u_k
    w = preference_vectors - current_pref               # (u_j - u_k) \forall j = 1, ..., K
    gx =  torch.matmul(w,losses/torch.norm(losses))     # In the paper they do not normalize the loss
    idx = gx >  0                                       # I(\theta), i.e the indexes of the active constraints
    
    active_constraints = w[idx]     # constrains which are violated, i.e. gx > 0

    # calculate the descent direction
    if torch.sum(idx) <= 0:
        flag = True
        return flag, torch.zeros(nobj)
    if torch.sum(idx) == 1:
        sol = torch.ones(1).cuda().float()
    else:
        # Equation 9
        # w[idx] = set of active constraints, i.e. where the solution is closer to another preference vector than the one desired.
        gx_gradient = torch.matmul(active_constraints, grads)    # We need to take the derivatives of G_j which is w.dot(grads)
        sol, nd = MinNormSolver.find_min_norm_element([[gx_gradient[t]] for t in range(len(gx_gradient))])
        sol = torch.Tensor(sol).cuda()
    
    # from MinNormSolver we get the weights (alpha) for each gradient. But we need the weights for the losses?
    weight = torch.matmul(sol, active_constraints)

    return flag, weight


def get_d_paretomtl(grads, losses, preference_vectors, pref_idx):
    """
    calculate the gradient direction for ParetoMTL 
    
    Args:
        grads: flattened gradients for each task
        losses: values of the losses for each task
        preference_vectors: all preference vectors u
        pref_idx: which index of u we are currently using
    """
    
    # check active constraints
    current_weight = preference_vectors[pref_idx]
    rest_weights = preference_vectors 
    w = rest_weights - current_weight
    
    gx =  torch.matmul(w,losses/torch.norm(losses))
    idx = gx >  0
    

    # calculate the descent direction
    if torch.sum(idx) <= 0:
        # here there are no active constrains in gx
        sol, nd = MinNormSolver.find_min_norm_element_FW([[grads[t]] for t in range(len(grads))])
        return torch.tensor(sol).cuda().float()
    else:
        # we have active constraints, i.e. we have move too far away from out preference vector
        #print('optim idx', idx)
        vec =  torch.cat((grads, torch.matmul(w[idx],grads)))
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])
        sol = torch.Tensor(sol).cuda()

        # FIX: handle more than just 2 objectives
        n = preference_vectors.shape[1]
        weights = []
        for i in range(n):
            weight_i =  sol[i] + torch.sum(torch.stack([sol[j] * w[idx][j - n , i] for j in torch.arange(n, n + torch.sum(idx))]))
            weights.append(weight_i)
        # weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,0] for j in torch.arange(2, 2 + torch.sum(idx))]))
        # weight1 =  sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,1] for j in torch.arange(2, 2 + torch.sum(idx))]))
        # weight = torch.stack([weight0,weight1])

        weight = torch.stack(weights)
        
        
        return weight


class ParetoMTLMethod(BaseMethod):

    def __init__(self, objectives, model, cfg):
        super().__init__(objectives, model, cfg)
        
        assert len(self.objectives) <= 2
        assert cfg.num_models == cfg.n_partitions + 1
        self.n = cfg.num_models

        # Ugly hack, we now needlesly train the main model, should be n-1 here.
        self.models = [deepcopy(model).cpu() for _ in range(self.n)]
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for m in self.models]
        self.schedulers = [get_lr_scheduler(cfg.lr_scheduler, o, cfg, '') for o in self.optimizers]
        self.init_sol_found = [False for _ in range(self.n)]

        self.pref_idx = -1
        # the original ref_vec can be obtained by circle_points(self.num_pareto_points, min_angle=0.0, max_angle=0.5 * np.pi)
        # we use the same min angle / max angle as for the other methods for comparison.
        self.ref_points = reference_points(self.n - 1)
        self.ref_vec = torch.Tensor(self.ref_points).to(cfg.device).float()

        self.eval_calls = 0


    def new_epoch(self, e):
        for m in self.models:
            m.train()
        if e>0:
            for s in self.schedulers:
                s.step()
        self.e = e
        self.eval_calls = 0


    def log(self):
        return {
            "train_ray": self.ref_vec[self.pref_idx].cpu().numpy().tolist(),
            "num_parameters": sum(num_parameters(m.parameters()) for m in self.models),
        }


    def _find_initial_solution(self, batch, model, pref_idx):

        grads = {}
        losses_vec = []
        
        # obtain and store the gradient value
        for i in self.task_ids:
            model.zero_grad()
            batch.update(model(batch))
            task_loss = self.objectives[i](**batch) 
            losses_vec.append(task_loss.data)
            
            task_loss.backward()
            
            grads[i] = []
            
            # can use scalable method proposed in the MOO-MTL paper for large scale problem
            # but we keep use the gradient of all parameters in this experiment
            for name, param in model.named_parameters():
                if param.grad is not None and param.grad.abs().sum() != 0:
                    grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

        
        grads_list = [torch.cat([g for g in grads[i]]) for i in grads]
        grads = torch.stack(grads_list)
        
        # calculate the weights
        losses_vec = torch.stack(losses_vec)
        init_solution_found, weight_vec = get_d_paretomtl_init(grads, losses_vec, self.ref_vec, pref_idx)
        


        if init_solution_found:
            print("Initial solution found")

        self.init_sol_found[pref_idx] = init_solution_found
        
        # optimization step
        model.zero_grad()
        batch.update(model(batch))
        
        loss_total = sum(w * self.objectives[i](**batch) for w, i in zip(weight_vec, self.task_ids))
        
        loss_total.backward()
        return loss_total.item()


    def step(self, batch):
        losses = []
        for idx, (optim, model) in enumerate(zip(self.optimizers, self.models)):
            model = model.to(self.device)
            model.train()

            result = self._step(batch, model, idx)
            optim.step()

            losses.append(result)
        return np.mean(losses).item()


    def _step(self, batch, model, pref_idx):

        if self.e < 2 and not self.init_sol_found[pref_idx]:
            # run at most 2 epochs to find the initial solution
            # stop early once a feasible solution is found 
            # usually can be found with a few steps
            return self._find_initial_solution(batch, model, pref_idx)
        else:
            # run normal update
            gradients, obj_values = calc_gradients(batch, model, self.objectives)
            
            grads = [torch.cat([torch.flatten(v) for k, v in sorted(gradients[t].items())]) for t in self.task_ids]
            grads = torch.stack(grads)
            
            # calculate the weights
            losses_vec = torch.Tensor([obj_values[t] for t in self.task_ids]).to(self.device)
            weight_vec = get_d_paretomtl(grads, losses_vec, self.ref_vec, pref_idx)
            
            normalize_coeff = len(self.objectives) / torch.sum(torch.abs(weight_vec))
            weight_vec = weight_vec * normalize_coeff
            
            # optimization step
            loss_total = None
            for a, t in zip(weight_vec, self.task_ids):
                logits = model(batch)
                batch.update(logits)
                task_loss = self.objectives[t](**batch)

                loss_total = a * task_loss if not loss_total else loss_total + a * task_loss
            
            loss_total.backward()
            return loss_total.item()
            
    
    def eval_step(self, batch, preference_vector):
        with torch.no_grad():
            model = self.models[self.eval_calls % self.n]   # will be called repeatedly for several batches
            model.eval()
            result = model(batch)
            self.eval_calls += 1
            return result
            
            
            # for i, ref_p in enumerate(self.ref_points):
            #     if all(ref_p == preference_vector):
            #         model = self.models[i].cuda()
            #         model.eval()
            #         result = model(batch)
            #         model.cpu()
            #         return result
        
        # raise ValueError("trying inference on different pref vec")
    

    def preference_at_inference(self):
        return True