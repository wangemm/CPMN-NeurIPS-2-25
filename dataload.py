# Please download dataset from: https://drive.google.com/file/d/1HGKDcyBXkjCBrkJlysWUFfVREm6_E8zc/view

import numpy as np
import scipy
import random


def data_load(args):
    data_path = '../Dataset/' + args.dataset + '/' + args.dataset + '.mat'
    data_mat = scipy.io.loadmat(data_path)
    if args.dataset == 'Caltech101':
        X = [data_mat['X'][0][0].T, data_mat['X'][0][1].T]
        GT = np.squeeze(data_mat['gt']) - 1
    elif args.dataset == 'Cub':
        X = [data_mat['X'][0][0].T, data_mat['X'][0][1].T]
        GT = np.squeeze(data_mat['gt']) - 1
    elif args.dataset == 'NoisyMNIST-30000':
        X = [data_mat['X1'], data_mat['X2']]
        GT = np.squeeze(data_mat['Y']) - 1
    elif args.dataset == 'NoisyMNIST':
        X = [data_mat['X1'], data_mat['X2']]
        GT = np.squeeze(data_mat['trainLabel']) - 1
    elif args.dataset == 'MNIST-USPS':
        X = [data_mat['X1'], data_mat['X2']]
        GT = np.squeeze(data_mat['Y'])
    elif args.dataset == 'Youtube-152549':
        X = [data_mat['gist'], data_mat['hist'], data_mat['hog']]  
        GT = np.squeeze(data_mat['label'])
    elif args.dataset == 'Reuters':
        X = [normalize(np.vstack((data_mat['x_train'][0], data_mat['x_test'][0]))),
             normalize(np.vstack((data_mat['x_train'][1], data_mat['x_test'][1])))]
        GT = np.squeeze(np.hstack((data_mat['y_train'], data_mat['y_test']))) - 1
    elif args.dataset == 'Scene15':
        X = [data_mat['X'][0][0], data_mat['X'][0][1]]
        GT = np.squeeze(data_mat['Y']) - 1
    elif args.dataset == '100Leaves':
        X = [data_mat['X'][0][0], data_mat['X'][0][1], data_mat['X'][0][2]]
        GT = np.squeeze(data_mat['Y']) - 1
    else:
        raise NotImplementedError('')
    return X, GT


def normalize(x):
    x = (x-np.tile(np.min(x, axis=0), (x.shape[0], 1))) / np.tile((np.max(x, axis=0)-np.min(x, axis=0)), (x.shape[0], 1))
    return x


def aligned_data_split(n_all, test_prop, seed):
    '''
    split data into training, testing dataset
    '''
    random.seed(seed)
    random_idx = random.sample(range(n_all), n_all)
    train_num = np.ceil((1-test_prop) * n_all).astype(int)
    train_idx = np.array(sorted(random_idx[0:train_num]))
    test_num = np.floor(test_prop * n_all).astype(int)
    test_idx = np.array(sorted(random_idx[-test_num:]))
    return train_idx, test_idx