import torch
import os
import numpy as np

from torch.utils import data

class MovieLens(data.Dataset):


    def __init__(self, root="data/movielens20m", split="train", **kwargs) -> None:
        super().__init__()

        if split == 'train':
            self.data = np.load(os.path.join(root, 'movielens_small_training.npy'))
            self.target = None
        elif split == 'val':
            self.data = np.load(os.path.join(root, 'movielens_small_validation_input.npy'))
            self.target = np.load(os.path.join(root, 'movielens_small_validation_test.npy'))
        elif split == 'test':
            self.data = np.load(os.path.join(root, 'movielens_small_test_input.npy'))
            self.target = np.load(os.path.join(root, 'movielens_small_test_test.npy'))
        else:
            raise ValueError(f"Unknown split {split}. Expected 'train', 'val', or 'test'.")
        
        self.data = torch.from_numpy(self.data)
        if self.target is not None:
            self.target = torch.from_numpy(self.target)
        
        print(f"Movielens 20m. Loaded {len(self)} instances for split {split}")
    

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index] if self.target is not None else self.data[index]     # TODO: understand why this is the case.
        return dict(data=x, labels=y)



