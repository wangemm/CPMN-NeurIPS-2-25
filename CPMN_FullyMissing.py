# Please download dataset from: https://drive.google.com/file/d/1HGKDcyBXkjCBrkJlysWUFfVREm6_E8zc/view

from __future__ import print_function, division
import argparse

import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.preprocessing import StandardScaler
import scipy.io
from idecutils import cluster_acc, target_distribution
from queue import Queue
from scipy.optimize import linear_sum_assignment
from losses import PCLoss, wmse_loss
from model import AEC_v2_m as AEC
from dataload import data_load
import os
import time
import sys

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def view_graph_match(yp0, yp1, cc0, cc1, args):
    ypredall = np.zeros(args.n_samples)
    u2v = {}
    v2u = {}

    cc0 = F.normalize(cc0, dim=1)
    cc1 = F.normalize(cc1, dim=1)
    similarity = ((torch.mm(cc0, cc1.T)) / 1).exp().cpu()
    cost = (1 / similarity)

    row_ind, col_ind = linear_sum_assignment(cost)
    for idx, item in enumerate(row_ind):
        v2u[row_ind[idx]] = col_ind[idx]
        u2v[col_ind[idx]] = row_ind[idx]

    ypredall[args.ind_0_complete] = yp0

    unique_values = np.unique(yp1)
    mapped_values = np.array([v2u[val] for val in unique_values])
    value_to_mapped = dict(zip(unique_values, mapped_values))
    mapped_vector = np.vectorize(value_to_mapped.get)(yp1)

    ypredall[args.ind_1_complete] = mapped_vector

    if np.any(ypredall is None):
        raise ValueError("Vector contains None values")

    return ypredall, u2v, v2u


def gcl_train(model, X0, X1, W, y, args):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.GCL_lr, weight_decay=args.weight_decay)
    pcloss = PCLoss(args)

    index_array = np.arange(X0.shape[0])
    np.random.shuffle(index_array)

    for epoch in range(args.GCL_epoch):
        total_loss = 0.
        for batch_idx in range(np.int_(np.ceil(X0.shape[0] / args.GCL_batch_size))):
            idx = index_array[
                  batch_idx * args.GCL_batch_size: min((batch_idx + 1) * args.GCL_batch_size, X0.shape[0])]
            x0 = X0[idx]
            x1 = X1[idx]
            w = W[idx]
            optimizer.zero_grad()

            x0_b, x1_b, _, _, z0_b, z1_b = model(x0, x1, w)

            # view-specific recons loss
            rec_loss = wmse_loss(x0_b, x0, w[:, 0]) + wmse_loss(x1_b, x1, w[:, 1])

            ze0 = z0_b[torch.where(w[:, 0] == 1)]
            ze1 = z1_b[torch.where(w[:, 1] == 1)]

            pc_loss = pcloss.graph_loss(ze0, ze1)

            fusion_loss = rec_loss + args.GCL_lam * pc_loss
            total_loss += fusion_loss
            fusion_loss.backward()
            optimizer.step()
        print('Epoch {}'.format(epoch), 'total_loss {:.4f}'.format(total_loss))

        if epoch == 0 or (epoch + 1) % 10 == 0:
            with torch.no_grad():
                _, _, _, _, zv0, zv1 = model(X0, X1, W)
                zv0com = zv0[args.ind_0_complete]
                zv1com = zv1[args.ind_1_complete]
                ycom0 = y[args.ind_0_complete]
                ycom1 = y[args.ind_1_complete]
                if epoch == 0:
                    kmeans0 = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=20)
                    kmeans1 = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=20)

                y0_pred = kmeans0.fit_predict(zv0com.data.cpu().numpy())
                y1_pred = kmeans1.fit_predict(zv1com.data.cpu().numpy())
                cc0 = torch.tensor(kmeans0.cluster_centers_).to(args.device)
                cc1 = torch.tensor(kmeans1.cluster_centers_).to(args.device)
                acc_v0 = cluster_acc(ycom0, y0_pred)
                acc_v1 = cluster_acc(ycom1, y1_pred)
                nmi_v0 = nmi_score(ycom0, y0_pred)
                nmi_v1 = nmi_score(ycom1, y1_pred)
                ari_v0 = ari_score(ycom0, y0_pred)
                ari_v1 = ari_score(ycom1, y1_pred)

                print({"View-1": {"acc": acc_v0, "nmi": nmi_v0, "ari": ari_v0},
                       "View-2": {"acc": acc_v1, "nmi": nmi_v1, "ari": ari_v1}})

                ypredall, _, _ = view_graph_match(y0_pred, y1_pred, cc0, cc1, args)
                acc = cluster_acc(y, ypredall)
                nmi = nmi_score(y, ypredall)
                ari = ari_score(y, ypredall)
                print({"acc": acc, "nmi": nmi, "ari": ari})
                model.cluster_layer_v0.data = torch.tensor(kmeans0.cluster_centers_).to(args.device)
                model.cluster_layer_v1.data = torch.tensor(kmeans1.cluster_centers_).to(args.device)

        torch.save(model.state_dict(), args.GCL_path)


def pcl_train(model, X0, X1, W, y, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.PCL_lr, weight_decay=args.weight_decay)
    pcloss = PCLoss(args)

    with torch.no_grad():
        _, _, q0, q1, _, _ = model(X0, X1, W)

        y_pred0 = q0.cpu().numpy().argmax(1)
        y_pred1 = q1.cpu().numpy().argmax(1)
        cc0 = model.cluster_layer_v0.data
        cc1 = model.cluster_layer_v1.data

        ypredall, _, _ = view_graph_match(y_pred0, y_pred1, cc0, cc1, args)

        acc = cluster_acc(y, ypredall)
        nmi = nmi_score(y, ypredall)
        ari = ari_score(y, ypredall)
        print({"acc": acc, "nmi": nmi, "ari": ari})

        y_pred_last = ypredall
        del y_pred0, y_pred1, cc0, cc1

    best_acc2 = 0
    best_epoch = 0
    for epoch in range(int(args.PCL_epoch)):

        if epoch % args.k_update == 0:
            _, _, tmp_q0, tmp_q1, z0, z1 = model(X0, X1, W)
            tmp_q0 = tmp_q0.data
            tmp_q1 = tmp_q1.data
            p0 = target_distribution(tmp_q0)
            p1 = target_distribution(tmp_q1)

            p_all0 = torch.zeros(size=[X0.shape[0], args.n_clusters]).to(args.device)
            p_all1 = torch.zeros(size=[X0.shape[0], args.n_clusters]).to(args.device)

            p_all0[args.ind_0_complete] = p0
            p_all1[args.ind_1_complete] = p1

            y_pred0 = tmp_q0.cpu().numpy().argmax(1)
            y_pred1 = tmp_q1.cpu().numpy().argmax(1)
            cc0 = model.cluster_layer_v0.data
            cc1 = model.cluster_layer_v1.data

            ypredall, u2v, v2u = view_graph_match(y_pred0, y_pred1, cc0, cc1, args)

            acc = cluster_acc(y, ypredall)
            nmi = nmi_score(y, ypredall)
            ari = ari_score(y, ypredall)

            if acc is None:
                break

            if acc > best_acc2:
                best_acc2 = np.copy(acc)
                best_epoch = epoch
                torch.save(model.state_dict(), args.PCL_path)
                print("model saved to {}.".format(args.PCL_path))

            print('Iter {}'.format(epoch), ':aveAcc {:.4f}'.format(acc), ':best_aveAcc {:.4f}'.format(best_acc2),
                  'best_Iter {}'.format(best_epoch))
            # check stop criterion
            delta_y = np.sum(ypredall != y_pred_last).astype(np.float32) / y_pred_last.shape[0]
            y_pred0_last = np.copy(ypredall)
            if epoch > 20 and delta_y < args.tol:
                print('Training stopped: epoch=%d, delta_label=%.4f, tol=%.4f' % (epoch, delta_y, args.tol))
                break

        index_array = np.arange(X0.shape[0])
        for batch_idx in range(np.int_(np.ceil(X0.shape[0] / args.PCL_batch_size))):
            idx = index_array[
                  batch_idx * args.PCL_batch_size: min((batch_idx + 1) * args.PCL_batch_size, X0.shape[0])]
            x0 = X0[idx]
            x1 = X1[idx]
            w = W[idx]

            e_idx0 = idx[np.where(w[:, 0].data.cpu().numpy() == 1)]
            e_idx1 = idx[np.where(w[:, 1].data.cpu().numpy() == 1)]
            optimizer.zero_grad()

            _, _, q0, q1, z0, z1 = model(x0, x1, w)
            # clustering loss
            kl_loss0 = F.kl_div(q0.log(), p_all0[e_idx0], reduction='batchmean')
            kl_loss1 = F.kl_div(q1.log(), p_all1[e_idx1], reduction='batchmean')
            kl_loss = kl_loss0 + kl_loss1
            # centres contrastive loss
            ze0 = z0[torch.where(w[:, 0] == 1)]
            ze1 = z1[torch.where(w[:, 1] == 1)]

            cc0 = model.cluster_layer_v0.data
            cc1 = model.cluster_layer_v1.data

            yp0 = q0.argmax(1)
            yp1 = q1.argmax(1)

            yp0_1 = torch.empty_like(yp0)
            for i in range(yp0.size(0)):
                yp0_1[i] = v2u.get(yp0[i].item(), yp0[i])

            yp1_0 = torch.empty_like(yp1)
            for i in range(yp1.size(0)):
                yp1_0[i] = u2v.get(yp1[i].item(), yp1[i])

            pcloss0 = pcloss.prototype_loss(ze0, cc0, cc1, yp0, yp0_1)
            pcloss1 = pcloss.prototype_loss(ze1, cc1, cc0, yp1, yp1_0)
            closs = pcloss0 + pcloss1

            fusion_loss = kl_loss + args.PCL_lam * closs
            fusion_loss.backward()
            optimizer.step()


def main():
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    if args.cuda:
        args.device_use = "cuda:" + str(args.device_num)
    args.device = torch.device(args.device_use if args.cuda else "cpu")
    print("USE {}".format(args.device))

    if args.seed is not None:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

    ####################################################################
    # Load data, label, incomplete_index_matrix
    ####################################################################
    X, GT = data_load(args)

    args.n_clusters = len(np.unique(GT))
    args.n_views = len(X)
    args.n_samples = len(X[0])

    dropMatrix = scipy.io.loadmat('./Data/' + args.dataset + '/' + args.dataset + '_' + str(args.full_incomplete) + '-FullMissing' + '.mat')

    if args.drop_index is None:
        dM = dropMatrix['IN_Index']
    else:
        dM = dropMatrix['IN_Index'][args.drop_index]

    if args.full_incomplete:
        args.aligned_p = 1

    args.basis_path = "./SaveWeight/" + args.dataset + "/MissingRate_" + str(args.aligned_p)

    if not os.path.exists(args.basis_path):
        os.makedirs(args.basis_path)

    args.GCL_path = (args.basis_path + '/GCL_Seed-' + str(args.seed) + '_zdim-' + str(args.z_dim) + '_GCL-BS-' + str(
        args.GCL_batch_size) + '_GCLepoch-' + str(args.GCL_epoch) + '_GCLlr-' + str(args.GCL_lr) + '_GCLlambda-' +
                      str(args.GCL_lam) + '_topk-' + str(args.topk) + '.pkl')

    args.PCL_path = (args.basis_path + '/GCL_Seed-' + str(args.seed) + '_zdim-' + str(args.z_dim) + '_GCL-BS-' + str(
        args.GCL_batch_size) + '_GCLepoch-' + str(args.GCL_epoch) + '_GCLlr-' + str(args.GCL_lr) + '_GCLlambda-' +
                      str(args.GCL_lam) + '_PCL-BS-' + str(args.PCL_batch_size) + '_PCLepoch-' +
                      str(args.PCL_epoch) + '_PCLlr-' + str(args.PCL_lr) + '_PCLlambda-' +
                      str(args.PCL_lam) + '.pkl')

    del dropMatrix
    ####################################################################
    # Preprocessing incomplete data
    ####################################################################
    # view-specific data
    X0 = np.array(X[0], 'float64')
    X1 = np.array(X[1], 'float64')
    args.n_input = [X0.shape[1], X1.shape[1]]

    # For each view,
    iv = 0
    # obtain drop and exist array
    WEiv = np.copy(dM[:, iv])
    args.ind_0_complete = np.where(WEiv == 1)
    args.ind_0_complete = (np.array(args.ind_0_complete)).reshape(-1)
    args.ind_0_dropped = np.where(WEiv == 0)
    args.ind_0_dropped = (np.array(args.ind_0_dropped)).reshape(-1)
    # normalize
    X0[args.ind_0_complete, :] = StandardScaler().fit_transform(X0[args.ind_0_complete, :])
    X0[args.ind_0_dropped, :] = 0

    iv = 1
    # obtain drop and exist array
    WEiv = np.copy(dM[:, iv])
    args.ind_1_complete = np.where(WEiv == 1)
    args.ind_1_complete = (np.array(args.ind_1_complete)).reshape(-1)
    args.ind_1_dropped = np.where(WEiv == 0)
    args.ind_1_dropped = (np.array(args.ind_1_dropped)).reshape(-1)
    # normalize
    X1[args.ind_1_complete, :] = StandardScaler().fit_transform(X1[args.ind_1_complete, :])
    X1[args.ind_1_dropped, :] = 0

    del X, iv, WEiv

    ##################################################################################
    # TrainProcess-1: GCL
    ##################################################################################
    X0 = torch.Tensor(np.nan_to_num(X0)).to(args.device)
    X1 = torch.Tensor(np.nan_to_num(X1)).to(args.device)
    W = torch.Tensor(dM).to(args.device)
    model = AEC(
        n_layers=args.layers_mlp,
        n_input=args.n_input,
        n_z=args.z_dim,
        n_h=args.n_clusters).to(args.device)

    if args.gcl_train_flag:
        gcl_train(model, X0, X1, W, GT, args)
        print('gcl_trained ae finished')
        args.gcl_train_flag = False
    else:
        model.load_state_dict(torch.load(args.GCL_path))
        print('load gcl_trained ae model from', args.GCL_path)

    #################################################################################
    # TrainProcess-2: PCL
    #################################################################################
    if args.pcl_train_flag:
        model.load_state_dict(torch.load(args.GCL_path))
        pcl_train(model, X0, X1, W, GT, args)
        print('pcl_trained ae finished')
        args.pcl_train_flag = False
    else:
        model.load_state_dict(torch.load(args.PCL_path))
        print('load pcl_trained ae model from', args.PCL_path)

    with (torch.no_grad()):
        model.load_state_dict(torch.load(args.PCL_path))
        _, _, Q0, Q1, _, _ = model(X0, X1, W)
        YP0 = Q0.cpu().numpy().argmax(1)
        YP1 = Q1.cpu().numpy().argmax(1)
        CC0 = model.cluster_layer_v0.data
        CC1 = model.cluster_layer_v1.data

        acc0 = cluster_acc(GT[args.ind_0_complete], YP0)
        acc1 = cluster_acc(GT[args.ind_1_complete], YP1)
        nmi0 = nmi_score(GT[args.ind_0_complete], YP0)
        nmi1 = nmi_score(GT[args.ind_1_complete], YP1)
        ari0 = ari_score(GT[args.ind_0_complete], YP0)
        ari1 = ari_score(GT[args.ind_1_complete], YP1)
        print({"View-1": {"acc": acc0, "nmi": nmi0, "ari": ari0},
               "View-2": {"acc": acc1, "nmi": nmi1, "ari": ari1}})

        ypredall, _, _ = view_graph_match(YP0, YP1, CC0, CC1, args)

        acc = cluster_acc(GT, ypredall)
        nmi = nmi_score(GT, ypredall)
        ari = ari_score(GT, ypredall)
        print({"acc": acc, "nmi": nmi, "ari": ari})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--full_incomplete', default=True, type=bool)
    parser.add_argument('--drop_index', default=0, type=int)
    parser.add_argument('--dataset', type=str, default='Caltech101')
    parser.add_argument('--basis_path', type=str, default='save_weight/Caltech101/')
    # Training Process
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--z_dim', default=32, type=int)  # 32 #
    parser.add_argument('--gcl_train_flag', default=True, type=bool)
    parser.add_argument('--pcl_train_flag', default=True, type=bool)
    # GCL_Train
    parser.add_argument('--topk', default=5, type=int)  # 5 #
    parser.add_argument('--GCL_epoch', default=200, type=int)  # 200 #
    parser.add_argument('--device_num', default=1, type=int)
    parser.add_argument('--GCL_lr', default=0.0001, type=float)  # 0.0001 #
    parser.add_argument('--GCL_batch_size', default=256, type=int)  # 256 #
    parser.add_argument('--GCL_lam', default=0.01, type=float)  # 0.01 #
    parser.add_argument('--t', default=1, type=float)
    parser.add_argument('--normalize_loss', default=True, type=bool)
    # PCL_Train
    parser.add_argument('--PCL_epoch', default=100, type=int)  # 200 #
    parser.add_argument('--k_update', default=2, type=int)   # 5 #
    parser.add_argument('--PCL_lr', default=0.0001, type=float)  # 0.0001 #
    parser.add_argument('--PCL_batch_size', default=256, type=int)  # 256 #
    parser.add_argument('--PCL_lam', default=0.01, type=float)  # 0.01 #
    # Fixed
    parser.add_argument('--tol', default=1e-7, type=float)
    parser.add_argument('--layers_mlp', default=4, type=int)
    parser.add_argument('--weight_decay', type=float, default=0., help='Initializing weight decay.')
    main()
