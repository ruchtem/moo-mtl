import torch
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt

from min_norm_solvers import MinNormSolver
from utils import is_pareto_efficient

factor = .01

def f1(x1, x2):
    return .5 * x1 + .5 * x2
    #return factor * ((x1-1)**2 + (x2-1)**2)

def f2(x1, x2):
    return -.5 * x1 + .5 * x2
    # return factor * ((x1-99)**2 + (x2-50)**2)

def f3(x1, x2):
    return 0 * x1 - np.sqrt(.5) * x2
    # return factor * ((x1-50)**2 + (x2-99)**2)

def loss(y, y_hat):
    return sum((y-y_hat)**2)


def gradient(X, y, w):
    dw1 = torch.sum(2* (y - torch.matmul(X, w)) * -X[:, 0].reshape(-1, 1), dim=0)
    dw2 = torch.sum(2* (y - torch.matmul(X, w)) * -X[:, 1].reshape(-1, 1), dim=0)
    return torch.stack((dw1, dw2))


def network(x, w):
    return torch.matmul(x, w)


class Synthetic(data.Dataset):

    def __init__(self, size=1000, std=5):

        self.X = torch.distributions.Uniform(-10, 10).sample((size, 2))
        self.y1 = f1(self.X[:, 0], self.X[:, 1]) #+ torch.distributions.Normal(loc=0, scale=std).sample((size, ))
        self.y2 = f2(self.X[:, 0], self.X[:, 1]) #+ torch.distributions.Normal(loc=0, scale=std).sample((size, ))
        self.y3 = f3(self.X[:, 0], self.X[:, 1]) #+ torch.distributions.Normal(loc=0, scale=std).sample((size, ))

        # pareto front in loss space
        w = []
        for w1 in np.linspace(-1, 1, 50):
            for w2 in np.linspace(-1, 1, 50):
                if w1 >= -.5 and w1 <= .5 and w2 >= -np.sqrt(.5) and w2 <= .5:
                    w.append(np.array([w1, w2]))

        self.loss_pf = self._loss(torch.Tensor(w))
        self.loss_buffer = []


    def _loss(self, w):
        assert w.ndim == 2
        y_hats = network(self.X, w.T)
        losses = [torch.mean((y_hats - y.reshape(-1, 1))**2, dim=0) for y in [self.y1, self.y2, self.y3]]
        return torch.stack(losses).T


    def __len__(self):
        """__len__"""
        return len(self.X)
    

    def __getitem__(self, index):
        return dict(data=self.X[index], labels1=self.y1[index], labels2=self.y2[index], labels3=self.y3[index])


    def getall(self):
        return dict(data=self.X, labels1=self.y1, labels2=self.y2, labels3=self.y3)


    def plot_pareto_loss(self, w=None):
        if w is not None:
            assert w.ndim == 2
            loss = self._loss(w)
            self.loss_buffer.append(loss)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=16., azim=41)
        ax.set_xlabel('loss 1')
        ax.set_ylabel('loss 2')
        ax.set_zlabel('loss 3')

        ax.plot(self.loss_pf[:, 0], self.loss_pf[:, 1], self.loss_pf[:, 2], '.')
        for loss in self.loss_buffer:
            ax.plot(loss[:, 0], loss[:, 1], loss[:, 2], 'r.')
        plt.savefig('u.png')
        #plt.show()
        plt.close()


    def plot_pareto_front(self):

        # Use this code to validate the pareto front
        # num_weights = 10000
        # weights = torch.distributions.Uniform(-.8, .6).sample((num_weights, 2))

        # y_hats = network(self.X, weights.T)

        # losses = [torch.sum((y_hats - y.reshape(-1, 1))**2, dim=0) for y in [self.y1, self.y2, self.y3]]
        # losses = torch.stack(losses).T

        # gradients = [gradient(self.X, y.reshape(-1, 1), weights.T) for y in [self.y1, self.y2, self.y3]]

        # min_norms = []
        # for g in range(num_weights):
        #     _, nd = MinNormSolver.scipy_impl([gradients[i][:, g] for i in range(3)])
        #     min_norms.append(nd)

        # idx = np.array(min_norms) < .1

        # plt.plot(weights[:, 0], weights[:, 1], 'ko')
        # plt.plot(weights[idx, 0], weights[idx, 1], 'ro')

        points = np.array([[.5, .5], [-.5, .5], [0, -np.sqrt(.5)], [.5, .5]])
        plt.plot(points[:, 0], points[:, 1], 'r-')


if __name__ == "__main__":
    dataset = Synthetic()

    dataset.plot_pareto_front()