import numpy as np

import torch
import torch.utils.data
from torch.autograd import Variable




from min_norm_solvers import MinNormSolver

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import CrossEntropyLoss


class RegressionTrain(torch.nn.Module):
  
    def __init__(self, model,init_weight):
        super(RegressionTrain, self).__init__()
        
        self.model = model
        self.weights = torch.nn.Parameter(torch.from_numpy(init_weight).float())
        self.ce_loss = CrossEntropyLoss()
    
    def forward(self, x, ts):
        n_tasks = 2
        ys = self.model(x)
     
        task_loss = []
        for i in range(n_tasks):
            task_loss.append( self.ce_loss(ys[:,i], ts[:,i]) )
        task_loss = torch.stack(task_loss)

        return task_loss


class RegressionModel(torch.nn.Module):
    def __init__(self, n_tasks):
        super(RegressionModel, self).__init__()
        self.n_tasks = n_tasks
        self.conv1 = nn.Conv2d(1, 10, 9, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.fc1 = nn.Linear(5*5*20, 50)
        
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*20)
        x = F.relu(self.fc1(x))
        
      
        outs = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            outs.append(layer(x))

        return torch.stack(outs, dim=1)


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


def circle_points(r, n):
    """
    generate evenly distributed unit preference vectors for two tasks
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles

from torch.utils import data
from loaders.multi_mnist_loader import MNIST
from datasets import global_transformer


npref = 5   

def train(pref_idx):
    # generate #npref preference vectors   
    
    n_tasks = 2
    init_weight = np.array([0.5 , 0.5 ])
    ref_vec = torch.tensor(circle_points([1], [npref])[0]).cuda().float()

    niter = 10

    # load dataset 

    # MultiMNIST: multi_mnist.pickle
    with open('data/mnist/multi_mnist.pickle','rb') as f:
        trainX, trainLabel,testX, testLabel = pickle.load(f)
    
    trainX = torch.from_numpy(trainX.reshape(120000,1,36,36)).float()
    trainLabel = torch.from_numpy(trainLabel).long()
    testX = torch.from_numpy(testX.reshape(20000,1,36,36)).float()
    testLabel = torch.from_numpy(testLabel).long()
    
    
    train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
    test_set  = torch.utils.data.TensorDataset(testX, testLabel)
    
    
    batch_size = 256
    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=batch_size,
                     shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)

    # train_set = MNIST("data/mnist", train=True, transform=global_transformer(), multi=True)
    # test_set = MNIST("data/mnist", train=False, transform=global_transformer(), multi=True)

    # num_workers = 4
    # train_loader = data.DataLoader(train_set, batch_size=256, num_workers=num_workers)
    # test_loader = data.DataLoader(test_set, batch_size=256, num_workers=num_workers)


    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader))) 


    # define the base model for ParetoMTL  
    model = RegressionTrain(RegressionModel(n_tasks), init_weight)



    if torch.cuda.is_available():
        model.cuda()


    # choose different optimizer for different base model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,45,60,75,90], gamma=0.5)




    # store infomation during optimization
    weights = []
    task_train_losses = []
    train_accs = []


    # print the current preference vector
    print('Preference Vector ({}/{}):'.format(pref_idx + 1, npref))
    print(ref_vec[pref_idx].cpu().numpy())

    # run at most 2 epochs to find the initial solution
    # stop early once a feasible solution is found 
    # usually can be found with a few steps
    for t in range(2):
        
        model.train()
        for (it, batch) in enumerate(train_loader):
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            grads = {}
            losses_vec = []
            
            
            # obtain and store the gradient value
            for i in range(n_tasks):
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
            flag, weight_vec = get_d_paretomtl_init(grads,losses_vec,ref_vec,pref_idx)
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
                
        else:
        # continue if no feasible solution is found
            continue
        # break the loop once a feasible solutions is found
        break
                
        

    # run niter epochs of ParetoMTL 
    for t in range(niter):
        
        scheduler.step()
        
        model.train()
        for (it, batch) in enumerate(train_loader):
            
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            # obtain and store the gradient 
            grads = {}
            losses_vec = []
            
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(X, ts) 
                losses_vec.append(task_loss[i].data)
                
                task_loss[i].backward()
            
                # can use scalable method proposed in the MOO-MTL paper for large scale problem
                # but we keep use the gradient of all parameters in this experiment              
                grads[i] = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

                
                
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
            optimizer.step()


        # calculate and record performance
        if t == 0 or (t + 1) % 2 == 0:
            
            model.eval()
            with torch.no_grad():

                total_train_loss = []
                train_acc = []
        
                correct1_train = 0
                correct2_train = 0
                
                for (it, batch) in enumerate(train_loader):
                    
                    X = batch[0]
                    ts = batch[1]
                    if torch.cuda.is_available():
                        X = X.cuda()
                        ts = ts.cuda()
        
                    valid_train_loss = model(X, ts)
                    total_train_loss.append(valid_train_loss)
                    output1 = model.model(X).max(2, keepdim=True)[1][:,0]
                    output2 = model.model(X).max(2, keepdim=True)[1][:,1]
                    correct1_train += output1.eq(ts[:,0].view_as(output1)).sum().item()
                    correct2_train += output2.eq(ts[:,1].view_as(output2)).sum().item()
                    
                    
                train_acc = np.stack([1.0 * correct1_train / len(train_loader.dataset),1.0 * correct2_train / len(train_loader.dataset)])
        
                total_train_loss = torch.stack(total_train_loss)
                average_train_loss = torch.mean(total_train_loss, dim = 0)
                
            
            # record and print
            if torch.cuda.is_available():
                
                task_train_losses.append(average_train_loss.data.cpu().numpy())
                train_accs.append(train_acc)
                
                weights.append(weight_vec.cpu().numpy())
                
                print('{}/{}: weights={}, train_loss={}, train_acc={}'.format(
                        t + 1, niter,  weights[-1], task_train_losses[-1],train_accs[-1]))                 
                

    torch.save(model.model.state_dict(), './saved_model/%s_%s_niter_%d_npref_%d_prefidx_%d.pickle'%(dataset, base_model, niter, npref, pref_idx))


for i in range(npref):
    train(pref_idx=i)