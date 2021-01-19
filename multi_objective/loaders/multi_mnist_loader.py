#Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py

from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import scipy.misc as m

from torchvision import transforms


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        split (str, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    multi_training_file = 'multi_training.pt'
    multi_test_file = 'multi_test.pt'

    data_buffer = None
    label_buffer = None

    def __init__(self, root='data/mnist', split='train', transform=transforms.ToTensor(), download=False, train_size=50000, multi=False):
        self.multi = multi
        self.split = split
        self.root = os.path.expanduser(root)
        self.transform = transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if not self._check_multi_exists():
            raise RuntimeError('Multi Task extension not found.' +
                               ' You can use download=True to download it')

        if split == 'test' or (MNIST.data_buffer is None or MNIST.label_buffer is None):
            self._load_into_buffer()
            

        if split == 'train':
            self.data = MNIST.data_buffer[:train_size].clone()
            self.labels = MNIST.label_buffer[:train_size].clone()
        elif split == 'val':
            self.data = MNIST.data_buffer[train_size:].clone()
            self.labels = MNIST.label_buffer[train_size:].clone()
        elif split == 'test':
            self.data = MNIST.data_buffer.clone()
            self.labels = MNIST.label_buffer.clone()
        else:
            raise ValueError("Unkown split {}, expecting either 'train', 'val', or 'test".format(split))


    def __getitem__(self, index):
        if not self.multi:
            return dict(data=self.data[index], labels=self.labels[index])
        else:
            return dict(data=self.data[index], label_l=self.labels[index, 0], label_r=self.labels[index, 1])


    def __len__(self):
        return len(self.labels)
    

    def getall(self):
        if not self.multi:
            return dict(data=self.data, labels=self.labels)
        else:
            return dict(data=self.data, label_l=self.labels[:, 0], label_r=self.labels[:, 1])


    def label_names(self):
        return ['label_l', 'label_r']


    def _load_into_buffer(self):
        if not self.multi:
            file = self.training_file if self.split in ['train', 'val'] else self.test_file
            data, labels = torch.load(os.path.join(self.root, self.processed_folder, file))
        else:
            file = self.multi_training_file if self.split in ['train', 'val'] else self.multi_test_file
            data, labels_l, labels_r = torch.load(os.path.join(self.root, self.processed_folder, file))
            labels = torch.vstack((labels_l, labels_r)).T
        
        # normalize
        data -= data.min(1, keepdim=True)[0]
        data /= data.max(1, keepdim=True)[0] + 1e-15

        # randomly shuffle and buffer
        idx = torch.randperm(data.shape[0])
        MNIST.data_buffer = data[idx].float()
        MNIST.label_buffer = labels[idx]

        MNIST.data_buffer = torch.unsqueeze(MNIST.data_buffer, dim=1)


    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))


    def _check_multi_exists(self):
        return  os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_test_file))


    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists() and self._check_multi_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')
        mnist_ims, multi_mnist_ims, extension = read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte'))
        mnist_labels, multi_mnist_labels_l, multi_mnist_labels_r = read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'), extension)

        tmnist_ims, tmulti_mnist_ims, textension = read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte'))
        tmnist_labels, tmulti_mnist_labels_l, tmulti_mnist_labels_r = read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'), textension)


        mnist_training_set = (mnist_ims, mnist_labels)
        multi_mnist_training_set = (multi_mnist_ims, multi_mnist_labels_l, multi_mnist_labels_r)

        mnist_test_set = (tmnist_ims, tmnist_labels)
        multi_mnist_test_set = (tmulti_mnist_ims, tmulti_mnist_labels_l, tmulti_mnist_labels_r)

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(mnist_training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(mnist_test_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.multi_training_file), 'wb') as f:
            torch.save(multi_mnist_training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.multi_test_file), 'wb') as f:
            torch.save(multi_mnist_test_set, f)
        print('Done!')


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path, extension):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        multi_labels_l = np.zeros((1*length),dtype=np.long)
        multi_labels_r = np.zeros((1*length),dtype=np.long)
        for im_id in range(length):
            for rim in range(1):
                multi_labels_l[1*im_id+rim] = parsed[im_id]
                multi_labels_r[1*im_id+rim] = parsed[extension[1*im_id+rim]] 
        return torch.from_numpy(parsed).view(length).long(), torch.from_numpy(multi_labels_l).view(length*1).long(), torch.from_numpy(multi_labels_r).view(length*1).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        pv = parsed.reshape(length, num_rows, num_cols)
        multi_length = length * 1
        multi_data = np.zeros((1*length, num_rows, num_cols))
        extension = np.zeros(1*length, dtype=np.int32)
        for left in range(length):
            chosen_ones = np.random.permutation(length)[:1]
            extension[left*1:(left+1)*1] = chosen_ones
            for j, right in enumerate(chosen_ones):
                lim = pv[left,:,:]
                rim = pv[right,:,:]
                new_im = np.zeros((36,36))
                new_im[0:28,0:28] = lim
                new_im[6:34,6:34] = rim
                new_im[6:28,6:28] = np.maximum(lim[6:28,6:28], rim[0:22,0:22])
                multi_data_im = np.array(Image.fromarray(new_im).resize((28, 28)))    #m.imresize(new_im, (28, 28), interp='nearest')
                multi_data[left*1 + j,:,:] = multi_data_im
        return torch.from_numpy(parsed).view(length, num_rows, num_cols), torch.from_numpy(multi_data).view(length,num_rows, num_cols), extension

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dst = MNIST(multi=True)
    loader = torch.utils.data.DataLoader(dst, batch_size=10, shuffle=True, num_workers=0)
    for dat in loader:
        ims = dat['data'].view(10,28,28).numpy()

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


