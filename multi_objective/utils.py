import torch
import numpy as np
import matplotlib.pyplot as plt


def calc_devisor(train_loader, model, objectives):
    values = np.zeros(len(objectives))
    for batch in train_loader:
        data = batch.data.cuda()
        labels = batch.labels.cuda()
        sensible_attribute = batch.sensible_attribute.cuda() if hasattr(batch, 'sensible_attribute') else None
        
        logits = model(data)

        for i, objective in enumerate(objectives):
            values[i] += objective(logits, labels, model, sensible_attribute)

    divisor = values / min(values)
    print("devisor={}".format(divisor))
    return divisor


def reset_weights(model):
    if isinstance(model, torch.nn.Conv2d) or isinstance(model, torch.nn.Linear):
        model.reset_parameters()


def calc_gradients(data, labels, sensible_attribute, model, objectives):
    # store gradients and objective values
    gradients = []
    obj_values = []
    for i, objective in enumerate(objectives):
        # zero grad
        model.zero_grad()
        
        logits = model(data)

        output = objective(logits, labels, model, sensible_attribute)
        output.backward()
        
        obj_values.append(output.item())
        gradients.append({})
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                gradients[i][name] = param.grad.data.detach().clone()
    
    return gradients, obj_values


class ParetoFront():

    def __init__(self, labels):
        self.labels = labels
        self.points = []

    def append(self, point):
        self.points.append(point)
    
    @staticmethod
    def _is_pareto_efficient(costs, return_mask=True):
        """
        From https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index<len(costs):
            nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype = bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient
    
    def plot(self):
        p = np.array(self.points)
        plt.plot(p[:, 0], p[:, 1], 'o')
        for i, text in enumerate(range(len(self.points))):
            plt.annotate(text, (p[i,0], p[i,1]))
        
        if p.shape[0] >= 3:
            front = p[self._is_pareto_efficient(p)]
            plt.plot(front[:, 0], front[:, 1], 'ro')

        plt.xlabel(self.labels[0])
        plt.ylabel(self.labels[1])
        plt.savefig("t.png")
        #plt.close()