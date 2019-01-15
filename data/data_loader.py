import sys
sys.path.insert(0, '../utils/')
import numpy as np
from bin_data import bin_single_router_pair

class loader(object):

    tr_id = [[10,  9,  2, 10,  4,  1,  3,  0,  2,  0,  3,  7,  1,  4,  4,  5,  6,  6, 2,  0],
             [ 7,  6,  5, 11,  9,  2,  4,  3,  0,  1,  8,  8,  7,  7,  3,  2,  7, 10, 6,  5]]
    te_id = [[ 8, 11,  4,  9,  0,],
             [ 6,  5,  8,  8,  0,]]
    va_id = [[ 1,  5, 10,  6,  5,],
             [ 8,  6,  8,  9,  1,]]

    def __init__(self, shuffle, bin_size, batch_size, split, mu=None, sigma=None):
        """
            Args:
                shuffle (bool):
                bin_size (int):
                batch_size (int):
                split (str): either 'tr', 'va' or 'te'
                mu (2-tuple(floats)): (mu_x, mu_y) from the train split.
                sigma (2-tuple(floats)): (sigma_x, sigma_y) from the train split.
        """
        self.shuffle = shuffle
        self.bin_size = bin_size
        self.batch_size = batch_size
        self.mu = mu
        self.sigma = sigma

        if split == 'tr':
            data_id = self.tr_id
        elif split == 'va':
            data_id = self.va_id
        else:
            data_id = self.te_id

        data_path = "../data/traffic_mats.npy"
        traffic_mats = np.load(data_path)[0:48000, :, :]

        self.x = []
        self.y = []
        for i in range(len(data_id[0])):
            # _x : (len(xs)/max_bin_size, max_bin_size)
            # _y : (len(xs)/max_bin_size,)
            _x, _y = bin_single_router_pair(traffic_mats, (data_id[0][i], data_id[1][i]))
            self.x.append(_x)
            self.y.append(_y)
        self.x = np.vstack(self.x)
        self.y = np.hstack(self.y)

        self.x = self.x[:, -bin_size:]

        # Only use mu, and sigma from the training set for the
        # validation or test set normalizations.
        if self.mu is None:
            self.mu = (np.mean(self.x, axis=0), np.mean(self.y))
            self.sigma = (np.std(self.x, axis=0), np.std(self.y))

        # Z-score X, and y to have mean 0, unit variance
        self.x = (self.x - self.mu[0]) / (self.sigma[0])
        self.y = (self.y - self.mu[1]) / (self.sigma[1])
        self.num_batches = self.x.shape[0] // self.batch_size
        self.num_samples = self.num_batches * self.batch_size
        self.step = 0

        if self.x.shape[0] % batch_size != 0:
            print("Warning: There are {} extra samples.\n\
            Take care for test and validation sets".format(
                self.x.shape[0] % batch_size))

    def load(self):
        if self.shuffle:
            if self.step % self.num_batches == 0:
                index  = np.arange(self.x.shape[0])
                np.random.shuffle(index)
                self.x = self.x[index]
                self.y = self.y[index]

        index_start = int((self.step * self.batch_size) % self.num_samples)
        index_end = int((index_start + self.batch_size))
        self.step += 1

        return (self.x[index_start:index_end].copy(),
                self.y[index_start:index_end].copy())

if __name__ == "__main__":
    train_loader = loader(True, 20, 128, 'tr')
    test_loader = loader(True, 20, 128, 'te', train_loader.mu, train_loader.sigma)
    print(train_loader.load()[0].shape)
