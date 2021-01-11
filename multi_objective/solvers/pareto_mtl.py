import torch
from torch.autograd import Variable


from utils import calc_gradients


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
        print(nd)
    
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
        sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
        return torch.tensor(sol).cuda().float()
    else:
        # we have active constraints
        vec =  torch.cat((grads, torch.matmul(w[idx],grads)))
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])


        weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,0] for j in torch.arange(2, 2 + torch.sum(idx))]))
        weight1 =  sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,1] for j in torch.arange(2, 2 + torch.sum(idx))]))
        weight = torch.stack([weight0,weight1])
        
        return weight


class ParetoMTLSolver():

    def __init__(self, objectives, model, reference_vector):
        self.objectives = objectives
        self.model = model
        self.train_loader = train_loader
        self.pref_idx = 0
        self.ref_vec = reference_vector


    def new_point(self, train_loader, optimizer):
        # run at most 2 epochs to find the initial solution
        # stop early once a feasible solution is found 
        # usually can be found with a few steps
        for t in range(2):
            
            self.model.train()
            for (it, batch) in enumerate(train_loader):
                X = batch[0]
                ts = batch[1]
                if torch.cuda.is_available():
                    X = X.cuda()
                    ts = ts.cuda()

                grads = {}
                losses_vec = []
                
                # obtain and store the gradient value
                for i in range(len(self.objectives)):
                    optimizer.zero_grad()
                    task_loss = model(X, ts) 
                    losses_vec.append(task_loss[i].data)
                    
                    task_loss[i].backward()
                    
                    grads[i] = []
                    
                    # can use scalable method proposed in the MOO-MTL paper for large scale problem
                    # but we keep use the gradient of all parameters in this experiment
                    for param in model.parameters():
                        if param.grad is not None:
                            grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

                
                grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
                grads = torch.stack(grads_list)
                
                # calculate the weights
                losses_vec = torch.stack(losses_vec)
                flag, weight_vec = get_d_paretomtl_init(grads, losses_vec, self.ref_vec, self.pref_idx)
                print(weight_vec)
                
                # early stop once a feasible solution is obtained
                if flag == True:
                    print("feasible solution is obtained.")
                    break
                
                # optimization step
                optimizer.zero_grad()
                for i in range(len(task_loss)):
                    task_loss = model(X, ts)
                    if i == 0:
                        loss_total = weight_vec[i] * task_loss[i]
                    else:
                        loss_total = loss_total + weight_vec[i] * task_loss[i]
                
                loss_total.backward()
                optimizer.step()
        self.pref_idx += 1


    def step(self, data, labels):
        gradients, obj_values = calc_gradients(data, labels, self.model, self.objectives)

        grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
        grads = torch.stack(grads_list)
        
        # calculate the weights
        losses_vec = torch.stack(losses_vec)
        weight_vec = get_d_paretomtl(grads,losses_vec,ref_vec,pref_idx)
        
        normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
        weight_vec = weight_vec * normalize_coeff
        
        # optimization step
        optimizer.zero_grad()
        for i in range(len(task_loss)):
            task_loss = model(X, ts)
            if i == 0:
                loss_total = weight_vec[i] * task_loss[i]
            else:
                loss_total = loss_total + weight_vec[i] * task_loss[i]
        
        loss_total.backward()