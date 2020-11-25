import os
import pandas as pd
import numpy as np

from torch.utils import data


def load_dataset(path):
    data = pd.read_csv(path, header=None)
    data = data.dropna()
    # split into x and y
    last_ix = len(data.columns) - 1
    y1 = data[last_ix]
    y2 = data[0]
    X = data.drop(last_ix, axis=1).drop(last_ix -1, axis=1).drop(0, axis=1)
    # encode
    X = pd.get_dummies(X).values
    y1 = (y1.values == ' <=50K.').astype('int') if ' <=50K.' in y1.values else (y1.values == ' <=50K').astype('int')
    y2 = (y2.values < 40).astype('int')
    return X, y1, y2



class ADULT(data.Dataset):
    def __init__(self, root, split="train"):
        assert split in ["train", "test"]
        path = os.path.join(root, "adult.data" if split=="train" else "adult.test")
        self.X, self.y1, self.y2 = load_dataset(path)
        print("loaded {} instances. y1 positives={}, y2 positives={}".format(len(self), sum(self.y1), sum(self.y2)))


    def __len__(self):
        """__len__"""
        return len(self.X)
    

    def __getitem__(self, index):
        return [self.X[index], self.y1[index], self.y2[index]]



if __name__ == "__main__":
    dataset = ADULT("data/adult", split="train")
    trainloader = data.DataLoader(dataset, batch_size=256, num_workers=0)

    for i, data in enumerate(trainloader):
        break