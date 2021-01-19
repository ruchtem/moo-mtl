import torch
import os
import pickle

class Multi(torch.utils.data.Dataset):


    def __init__(self, dataset, split, root='data/multi'):
        assert dataset in ['mnist', 'fashion', 'fashion_and_mnist']
        assert split in ['train', 'val', 'test']

        train_split = 5/6

        # MultiMNIST: multi_mnist.pickle
        if dataset == 'mnist':
            with open(os.path.join(root, 'multi_mnist.pickle'),'rb') as f:
                trainX, trainLabel, testX, testLabel = pickle.load(f)  
        
        # MultiFashionMNIST: multi_fashion.pickle
        if dataset == 'fashion':
            with open(os.path.join(root, 'multi_fashion.pickle'),'rb') as f:
                trainX, trainLabel,testX, testLabel = pickle.load(f)  
        
        
        # Multi-(Fashion+MNIST): multi_fashion_and_mnist.pickle
        if dataset == 'fashion_and_mnist':
            with open(os.path.join(root, 'multi_fashion_and_mnist.pickle'),'rb') as f:
                trainX, trainLabel,testX, testLabel = pickle.load(f)
        
        trainX = torch.Tensor(trainX)
        trainLabel = torch.Tensor(trainLabel).long()
        testX = torch.Tensor(testX)
        testLabel = torch.Tensor(testLabel).int()

        # normalize
        trainX -= trainX.min(1, keepdim=True)[0]
        trainX /= trainX.max(1, keepdim=True)[0] + 1e-15
        testX -= testX.min(1, keepdim=True)[0]
        testX /= testX.max(1, keepdim=True)[0] + 1e-15

        # randomly shuffle and buffer
        idx = torch.randperm(trainX.shape[0])
        trainX = trainX[idx].float()
        trainLabel = trainLabel[idx]

        if split in ['train', 'val']:
            n = int(len(trainX) * train_split)
            if split == 'val':
                self.X = trainX[n:]
                self.y = trainLabel[n:]
            elif split == 'train':
                self.X = trainX[:n]
                self.y = trainLabel[:n]
        else:
            self.X = testX
            self.y = testLabel
        
        self.X = torch.unsqueeze(self.X, dim=1)

    
    def __getitem__(self, index):
        return dict(data=self.X[index], label_l=self.y[index, 0], label_r=self.y[index, 1])
    

    def __len__(self):
        return len(self.X)


    def label_names(self):
        return ['label_l', 'label_r']


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dst = Multi(dataset='fashion_and_mnist', split='val')
    loader = torch.utils.data.DataLoader(dst, batch_size=10, shuffle=True, num_workers=0)
    for dat in loader:
        ims = dat['data'].view(10,36,36).numpy()

        labs_l = dat['label_l']
        labs_r = dat['label_r']
        f, axarr = plt.subplots(2,5)
        for j in range(5):
            for i in range(2):
                axarr[i][j].imshow(ims[j*2+i,:,:], cmap='gray')
                axarr[i][j].set_title('{}_{}'.format(labs_l[j*2+i],labs_r[j*2+i]))
        plt.show()
        a = input()
        if a == 'ex':
            break
        else:
            plt.close()