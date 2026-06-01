import numpy as np
import scipy.io as scio
import argparse
import random
import math
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


def get_sn(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate: missing samples in each view
    :return:Sn
    """
    one_rate = 1.0 - (missing_rate / 100) / view_num
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()  # n_values=view_num
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix
    while error >= 0.0001:
        enc = OneHotEncoder()  # n_values=view_num
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    return matrix


def aligned_data_split(n_all, test_prop, seed):
    '''
    split data into training, testing dataset
    '''
    random.seed(seed)
    random_idx = random.sample(range(n_all), n_all)
    train_num = np.ceil((1 - test_prop) * n_all).astype(int)
    train_idx = np.array(sorted(random_idx[0:train_num]))
    test_num = np.floor(test_prop * n_all).astype(int)
    test_idx = np.array(sorted(random_idx[-test_num:]))
    return train_idx, test_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--FullMissing', type=bool, default=False,
                        help='True-FullMissing, False-PartialMissing')
    parser.add_argument('--dataset', type=str, default='Caltech101',
                        help='Cub, Caltech101, Reuters, Scene15, 100Leaves, NoisyMNIST-30000, NoisyMNIST, MNIST-USPS, Youtube-152549')
    # parser.add_argument('--incompleteRate', type=float, default=50)
    args = parser.parse_args()
    data_path = './Data/' + args.dataset + '/' + args.dataset + '.mat'
    data_mat = scio.loadmat(data_path)
    data = []

    if args.dataset == 'Scene15':
        data.append(data_mat['X'][0][0])
        data.append(data_mat['X'][0][1])
        label = np.squeeze(data_mat['Y']) - 1

    elif args.dataset == 'Caltech101':
        data.append(data_mat['X'][0][0].T)
        data.append(data_mat['X'][0][1].T)
        label = np.squeeze(data_mat['gt']) - 1

    elif args.dataset == 'Reuters':
        data.append(np.vstack((data_mat['x_train'][0], data_mat['x_test'][0])))
        data.append(np.vstack((data_mat['x_train'][1], data_mat['x_test'][1])))
        label = np.squeeze(np.hstack((data_mat['y_train'], data_mat['y_test'])))

    elif args.dataset == 'Cub':
        label = np.squeeze(data_mat['gt']) - 1
        data.append(data_mat['X'][0][0].T)
        data.append(data_mat['X'][0][1].T)

    elif args.dataset == '100Leaves':
        data.append(data_mat['X'][0][0])
        data.append(data_mat['X'][0][1])
        data.append(data_mat['X'][0][2])
        label = np.squeeze(data_mat['Y']) - 1

    elif args.dataset == 'NoisyMNIST-30000':
        data.append(data_mat['X1'])
        data.append(data_mat['X2'])
        label = np.squeeze(data_mat['Y']) - 1
    elif args.dataset == 'NoisyMNIST':
        data.append(data_mat['X1'])
        data.append(data_mat['X2'])
        label = np.squeeze(data_mat['trainLabel']) - 1
    elif args.dataset == 'MNIST-USPS':
        data.append(data_mat['X1'])
        data.append(data_mat['X2'])
        label = np.squeeze(data_mat['Y'])
    elif args.dataset == 'Youtube-152549':
        data.append(data_mat['gist'])  # 1024
        data.append(data_mat['hist'])  # 768
        data.append(data_mat['hog'])   # 1152
        label = np.squeeze(data_mat['label'])
    else:
        raise NotImplementedError('')

    views = len(data)
    print('Number of samples:', label.shape[0])
    print('Number of views:', views)

    Incomplete_Index = np.zeros((10, label.shape[0], views))
    if args.FullMissing:  # Full Incomplete Data
        args.IncompleteRate = 100
        for i in range(Incomplete_Index.shape[0]):
            mask = get_sn(views, label.shape[0], args.IncompleteRate)
            print(mask.sum() / label.shape[0])
            Incomplete_Index[i] = mask

    else: # Partial Incomplete Data
        args.IncompleteRate = 50
        for i in range(Incomplete_Index.shape[0]):
            mask = get_sn(views, label.shape[0], args.IncompleteRate)
            print(mask.sum() / label.shape[0])
            Incomplete_Index[i] = mask

    scio.savemat('./Data/' + args.dataset + '/' + args.dataset + '_' + str(args.FullMissing) + '-FullMissing' + '.mat', {'IN_Index': Incomplete_Index})
