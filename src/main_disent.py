'''
Disentangled multiple cause effect learning
2020-07-08
'''

import time
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import math
import argparse
import os
import sys
import scipy.io as scio
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as tsn
from scipy.stats import pearsonr

from data_synthetic import plot_cluster, generate_y_final, get_y_final
from model_disent import MTvae

sys.path.append('../')

font_sz = 28
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
matplotlib.rcParams.update({'font.size': font_sz})

parser = argparse.ArgumentParser(description='Disentangled multiple cause VAE')
parser.add_argument('--nocuda', type=int, default=0, help='Disables CUDA training.')
parser.add_argument('--batch-size', type=int, default=1500, metavar='N',
                    help='input batch size for training (default: 10000)')
parser.add_argument('--epochs', type=int, default=251, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--K', type=int, default=4, metavar='N',
                    help='number of clusters')
parser.add_argument('--trn_rate', type=float, default=0.6, help='training data ratio')
parser.add_argument('--tst_rate', type=float, default=0.2, help='test data ratio')
parser.add_argument('--mu_p_wt', type=float, default=1.0, help='weight for mu_p_t')

parser.add_argument('--dim_zt', type=int, default=32, metavar='N',
                    help='dimension of zt')
parser.add_argument('--dim_zi', type=int, default=32, metavar='N',
                    help='dimension of zi')
parser.add_argument('--nogb', action='store_true', default=False,
                    help='Disable Gumbel-Softmax sampling.')

parser.add_argument('--beta', type=float, default=20, help='weight for loss balance')

parser.add_argument('--dataset', default='synthetic', help='dataset to use')  # synthetic, amazon, amazon_6c
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate for optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight decay')


args = parser.parse_args()

# select gpu if available
args.cuda = not args.nocuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
args.device = device

print('using device: ', device)

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def loss_function(input_ins_batch, mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc, mu_zi_list, logvar_zi_list, zi_sample_list, a_pred, mu_y, logvar_y, target, a_reconstby_zt, input_treat_trn):
    # 1. recontrust loss
    loss_bce = nn.BCELoss(reduction='mean').to(device)
    loss_reconst = loss_bce(a_pred.view(-1), input_ins_batch.view(-1))

    # 2. KLD_C
    KLD_C = torch.mean(torch.sum(qc * torch.log(args.K * qc + 1e-10), dim=1), dim=0)

    # 3. E_KLD_QT_PT
    mu_zt = mu_zt.unsqueeze(-1)
    logvar_zt = logvar_zt.unsqueeze(-1)

    mu_p_zt = mu_p_zt.T
    logvar_p_zt = logvar_p_zt.T
    mu_p_zt = mu_p_zt.unsqueeze(0)
    logvar_p_zt = logvar_p_zt.unsqueeze(0)

    KLD_QT_PT = 0.5 * (((logvar_p_zt - logvar_zt) + ((logvar_zt.exp() + (mu_zt - args.mu_p_wt * mu_p_zt).pow(2)) / logvar_p_zt.exp())) - 1)

    # zt
    loss_bce2 = nn.BCELoss(reduction='mean').to(device)
    loss_reconst_zt = loss_bce2(a_reconstby_zt.reshape(-1), input_treat_trn.reshape(-1))

    qc = qc.unsqueeze(-1)  # m x k x 1
    qc = qc.expand(-1, args.K, 1)  # m x k x 1

    E_KLD_QT_PT = torch.mean(torch.sum(torch.bmm(KLD_QT_PT, qc), dim=1), dim=0)

    # 4. KL_ZI
    # KL_ZI = None
    # for k in range(args.K):
    #     mu_zi_k = mu_zi_list[k]  # batch_size x d
    #     logvar_zi_k = logvar_zi_list[k]
    #     kl_zi_k = -0.5 * torch.sum(1 + logvar_zi_k - mu_zi_k.pow(2) - logvar_zi_k.exp(), dim=1)  # n
    #     KL_ZI = (kl_zi_k if (KL_ZI is None) else (KL_ZI + kl_zi_k))  #
    # KL_ZI = torch.mean(KL_ZI, dim=0)

    #
    mu_zi_all = None
    log_zi_all = None
    for k in range(args.K):
        mu_zi_k = mu_zi_list[k]
        logvar_zi_k = logvar_zi_list[k]
        mu_zi_all = mu_zi_k if mu_zi_all is None else torch.cat([mu_zi_all, mu_zi_k], dim=1)
        log_zi_all = logvar_zi_k if log_zi_all is None else torch.cat([log_zi_all, logvar_zi_k], dim=1)
    KL_ZI = -0.5 * torch.sum(1 + log_zi_all - mu_zi_all.pow(2) - log_zi_all.exp(), dim=1)  # n
    KL_ZI = torch.mean(KL_ZI, dim=0)


    # 5. loss_y
    temp = 0.5 * math.log(2 * math.pi)
    target = target.view(-1, 1)

    bb = - 0.5 * ((target - mu_y).pow(2)) / logvar_y.exp() - 0.5 * logvar_y - temp
    loss_y = - torch.mean(torch.sum(- 0.5 * ((target - mu_y).pow(2)) / logvar_y.exp() - 0.5 * logvar_y - temp, dim=1), dim=0)

    # MSE_Y
    loss_mse = nn.MSELoss(reduction='mean')
    loss_y_mse = loss_mse(mu_y, target)

    # 6. loss balance
    loss_balance = 0.0

    loss = loss_reconst + KL_ZI + KLD_C + E_KLD_QT_PT + loss_y

    eval_result = {
        'loss': loss, 'loss_reconst': loss_reconst, 'KLD_C': KLD_C, 'E_KLD_QT_PT': E_KLD_QT_PT, 'loss_reconst_zt':loss_reconst_zt,
        'KL_ZI': KL_ZI, 'loss_y': loss_y, 'loss_y_mse': loss_y_mse, 'loss_balance': loss_balance,
    }

    return eval_result


def test(model, data_loader, input_treat_trn, adj_assign, Z_i_list, Zt, params, C_true, inx_spec_treat=None, show_cluster=None, show_disent=True, show_y=False):

    model.eval()

    num_cluster = args.K
    m = input_treat_trn.shape[0]
    num_assign = len(adj_assign)
    pehe = torch.zeros(num_assign, dtype = torch.float)

    ite_true_sum = torch.zeros(num_assign, dtype = torch.float)
    ite_pred_sum = torch.zeros(num_assign, dtype = torch.float)

    adj_pred_correctNum = 0.0  # m

    data_size = 0

    for batch_idx, (adj_batch, target, orin_index) in enumerate(data_loader):
        data_size += adj_batch.shape[0]
        batch_size = adj_batch.shape[0]

        if args.cuda:
            adj_batch = adj_batch.to(device)
            orin_index = orin_index.to(device)

        mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc, mu_zi_list, logvar_zi_list, zi_sample_list, a_pred, mu_y, logvar_y, a_reconstby_zt = model(
            adj_batch, input_treat_trn)

        # accuracy of treatment assignment prediction
        a_pred[a_pred >= 0.5] = 1.0
        a_pred[a_pred < 0.5] = 0.0

        if inx_spec_treat is None:
            adj_pred_correctNum += (a_pred == adj_batch).sum()
        else:
            a_pred_spec = a_pred[:, inx_spec_treat]
            a_true_spec = adj_batch[:, inx_spec_treat]
            adj_pred_correctNum += (a_pred_spec == a_true_spec).sum()

        # get true y
        if Z_i_list is None:
            y_true = torch.zeros((batch_size, len(adj_assign), 1), device=args.device)
            y_true_0 = torch.zeros((batch_size, 1), device=args.device)
        else:
            y_true, y_true_0 = get_y_true_final(orin_index, adj_assign, Z_i_list, Zt, params)

        # pehe, ate
        adj_batch_0 = torch.zeros([adj_batch.shape[0], m], dtype=torch.float)  # batch size x m

        if args.cuda:
            adj_batch_0 = adj_batch_0.to(device)

        y_pred_0, _ = model.predictY(mu_zt, zi_sample_list, qc, adj_batch_0)

        for j in range(len(adj_assign)):
            adj_assign_j = adj_assign[j]  # m
            adj_assign_j = adj_assign_j.unsqueeze(0)
            adj_assign_j = adj_assign_j.expand(adj_batch.shape[0], m)
            if args.cuda:
                adj_assign_j = adj_assign_j.to(device)

            y_pred_j, _ = model.predictY(mu_zt, zi_sample_list, qc, adj_assign_j)
            y_true_j = y_true[:, j, :]

            ite_pred_j = y_pred_j - y_pred_0
            ite_true_j = y_true_j - y_true_0

            pehe[j] = pehe[j] + torch.sum((ite_pred_j - ite_true_j).pow(2))

            ite_true_sum[j] = ite_true_sum[j] + torch.sum(ite_true_j)
            ite_pred_sum[j] = ite_pred_sum[j] + torch.sum(ite_pred_j)

    pehe = torch.sqrt(pehe / data_size)
    pehe_ave = torch.sum(pehe) / num_assign

    ate = torch.abs(ite_true_sum / data_size - ite_pred_sum / data_size)
    ate_ave = torch.sum(ate) / num_assign

    if inx_spec_treat is None:
        acc_apred = adj_pred_correctNum / (data_size * m)
    else:
        m_new = len(inx_spec_treat)
        acc_apred = adj_pred_correctNum / (data_size * m_new)

    # acc of zt
    a_reconstby_zt[a_reconstby_zt >= 0.5] = 1.0
    a_reconstby_zt[a_reconstby_zt < 0.5] = 0.0

    adj_pred_correctNum_zt = 0.0
    adj_pred_correctNum_zt += (a_reconstby_zt == input_treat_trn).sum()
    acc_apred_zt = adj_pred_correctNum_zt / (input_treat_trn.shape[0] * input_treat_trn.shape[1])

    if show_cluster is not None:
        C = torch.argmax(qc, dim=1).cpu().detach().numpy()  # m
        mu_zt = mu_zt.cpu().detach().numpy()
        mu_p_zt = args.mu_p_wt * mu_p_zt.cpu().detach().numpy()
        Zt_tsn = plot_cluster(mu_zt, C, num_cluster, mu_zt_all=mu_p_zt, saving=False)

        # true clusters
        plot_cluster(mu_zt, C_true, num_cluster, mu_zt_all=mu_p_zt, saving=False, Zt_tsn=Zt_tsn)

    eval_result = {
        'pehe': pehe_ave, 'ate': ate_ave, 'acc_apred': acc_apred,
        'acc_apred_zt': acc_apred_zt
    }

    return eval_result

def train(epochs, model, trn_loader, val_loader, tst_loader, input_treat_trn, adj_assign, Z_i_list, Zt, params, C_true, optimizer, with_test=True, active_opt=[True, True, True, True]):
    time_begin = time.time()

    model.train()
    print("start training!")

    optimizer_1 = optimizer[0]
    optimizer_2 = optimizer[1]
    optimizer_3 = optimizer[2]
    optimizer_4 = optimizer[3]

    for epoch in range(epochs):
        for batch_idx, (adj_batch, target, orin_index) in enumerate(trn_loader):
            if args.cuda:
                adj_batch = adj_batch.to(device)
                target = target.to(device)

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            optimizer_3.zero_grad()
            optimizer_4.zero_grad()

            # forward pass
            if active_opt[0]:
                for i in range(5):
                    optimizer_1.zero_grad()

                    mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc, mu_zi_list, logvar_zi_list, zi_sample_list, a_pred, mu_y, logvar_y, a_reconstby_zt = model(adj_batch, input_treat_trn)

                    eval_result = loss_function(adj_batch, mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc, mu_zi_list, logvar_zi_list, zi_sample_list, a_pred, mu_y, logvar_y, target, a_reconstby_zt,input_treat_trn)
                    loss, KLD_C, E_KLD_QT_PT, loss_a_reconst_zt, loss_reconst, KL_ZI, KLD_C, loss_y, loss_y_mse = \
                        eval_result['loss'], eval_result['KLD_C'], eval_result['E_KLD_QT_PT'], eval_result['loss_reconst_zt'], eval_result['loss_reconst'], eval_result['KL_ZI'], eval_result['KLD_C'], eval_result['loss_y'], eval_result['loss_y_mse']
                    # backward propagation
                    (loss_a_reconst_zt).backward()
                    optimizer_1.step()

            if active_opt[2]:
                for i in range(3):
                    optimizer_3.zero_grad()

                    mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc, mu_zi_list, logvar_zi_list, zi_sample_list, a_pred, mu_y, logvar_y, a_reconstby_zt = model(
                        adj_batch, input_treat_trn)

                    eval_result = loss_function(adj_batch, mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc, mu_zi_list,
                                                logvar_zi_list, zi_sample_list, a_pred, mu_y, logvar_y, target,
                                                a_reconstby_zt, input_treat_trn)
                    loss, KLD_C, E_KLD_QT_PT, loss_a_reconst_zt, loss_reconst, KL_ZI, KLD_C, loss_y, loss_y_mse = \
                        eval_result['loss'], eval_result['KLD_C'], eval_result['E_KLD_QT_PT'], eval_result[
                            'loss_reconst_zt'], eval_result['loss_reconst'], eval_result['KL_ZI'], eval_result['KLD_C'], \
                        eval_result['loss_y'], eval_result['loss_y_mse']

                    # backward propagation
                    pm_beta = 1.0 if epoch < 100 else args.beta
                    (loss_reconst + pm_beta * KL_ZI).backward()
                    optimizer_3.step()

            if active_opt[3]:
                for i in range(20):
                    optimizer_4.zero_grad()

                    mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc, mu_zi_list, logvar_zi_list, zi_sample_list, a_pred, mu_y, logvar_y, a_reconstby_zt = model(
                        adj_batch, input_treat_trn)

                    eval_result = loss_function(adj_batch, mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc, mu_zi_list,
                                                logvar_zi_list, zi_sample_list, a_pred, mu_y, logvar_y, target,
                                                a_reconstby_zt, input_treat_trn)
                    loss, KLD_C, E_KLD_QT_PT, loss_a_reconst_zt, loss_reconst, KL_ZI, KLD_C, loss_y, loss_y_mse = \
                        eval_result['loss'], eval_result['KLD_C'], eval_result['E_KLD_QT_PT'], eval_result[
                            'loss_reconst_zt'], eval_result['loss_reconst'], eval_result['KL_ZI'], eval_result['KLD_C'], \
                        eval_result['loss_y'], eval_result['loss_y_mse']

                    # backward propagation
                    loss_y.backward()
                    optimizer_4.step()

            # optimize for the centroid
            if active_opt[1]:
                for i in range(20):
                    optimizer_2.zero_grad()

                    # forward pass
                    mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc, mu_zi_list, logvar_zi_list, zi_sample_list, a_pred, mu_y, logvar_y, a_reconstby_zt = model(
                        adj_batch, input_treat_trn)

                    eval_result = loss_function(adj_batch, mu_zt, logvar_zt, mu_p_zt, logvar_p_zt, qc, mu_zi_list,
                                                logvar_zi_list, zi_sample_list, a_pred, mu_y, logvar_y, target,
                                                a_reconstby_zt, input_treat_trn)
                    loss, KLD_C, E_KLD_QT_PT, loss_a_reconst_zt, loss_reconst, KL_ZI, KLD_C, loss_y, loss_y_mse = \
                        eval_result['loss'], eval_result['KLD_C'], eval_result['E_KLD_QT_PT'], eval_result[
                            'loss_reconst_zt'], eval_result['loss_reconst'], eval_result['KL_ZI'], eval_result['KLD_C'], \
                        eval_result['loss_y'], eval_result['loss_y_mse']

                    # backward propagation
                    (5*KLD_C+E_KLD_QT_PT).backward()
                    optimizer_2.step()

        # evaluate
        if epoch % 100 == 0:
            show_disent = True
            model.eval()
            # eval_result_val = test(model, val_loader, input_treat_trn, adj_assign, Z_i_list, Zt, params, C_true)
            eval_result_tst = test(model, tst_loader, input_treat_trn, adj_assign, Z_i_list, Zt, params, C_true, show_disent=show_disent)
            pehe_tst, mae_ate_tst = eval_result_tst['pehe'], eval_result_tst['ate']
            print('Epoch: {:04d}'.format(epoch + 1),
                  'pehe_tst: {:.4f}'.format(pehe_tst.item()),
                  'mae_ate_tst: {:.4f}'.format(mae_ate_tst.item()),
                  'time: {:.4f}s'.format(time.time() - time_begin))
            model.train()

    return

class Synthetic_dataset(Dataset):
    def __init__(self, adj, y, trn_idx=None, val_idx=None, tst_idx=None, type='train'):
        n = adj.shape[0]

        if trn_idx is None:
            size_trn = int(args.trn_rate * n)
            size_tst = int(args.tst_rate * n)
            size_val = n - size_trn - size_tst

            if type == 'train':
                self.adj_ins = adj[:size_trn]
                self.target = y[:size_trn]
                self.orin_index = np.array(range(size_trn))
            elif type == 'val':
                self.adj_ins = adj[size_trn: size_trn + size_val]
                self.target = y[size_trn: size_trn + size_val]
                self.orin_index = np.array(range(size_trn, size_trn + size_val))
            elif type == 'test':
                self.adj_ins = adj[size_trn + size_val:]
                self.target = y[size_trn + size_val:]
                self.orin_index = np.array(range(size_trn + size_val, n))

        else:
            if type == 'train':
                self.adj_ins = adj[trn_idx]
                self.target = y[trn_idx]
                self.orin_index = trn_idx
            elif type == 'val':
                self.adj_ins = adj[val_idx]
                self.target = y[val_idx]
                self.orin_index = val_idx
            elif type == 'test':
                self.adj_ins = adj[tst_idx]
                self.target = y[tst_idx]
                self.orin_index = tst_idx

    def __getitem__(self, index):
        adj = self.adj_ins[index]
        target = self.target[index]
        orin_index = self.orin_index[index]
        return adj, target, orin_index

    def __len__(self):
        return len(self.adj_ins)


def Sythetic_loader(batchSize, adj, y, trn_idx=None, val_idx=None, tst_idx=None):
    adj = torch.FloatTensor(adj)
    y = torch.FloatTensor(y)

    train_loader = torch.utils.data.DataLoader(
        Synthetic_dataset(adj, y, trn_idx, val_idx, tst_idx, type='train'),
        batch_size=batchSize,
        shuffle=False
    )

    val_loader = torch.utils.data.DataLoader(
        Synthetic_dataset(adj, y, trn_idx, val_idx, tst_idx, type='val'),
        batch_size=batchSize,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        Synthetic_dataset(adj, y, trn_idx, val_idx, tst_idx, type='test'),
        batch_size=batchSize,
        shuffle=False
    )

    return train_loader, val_loader, test_loader

def get_y_true_final(orin_index, adj_assign, Z_i_list, Zt, params):
    batch_size = orin_index.shape[0]
    num_assign = adj_assign.shape[0]
    m = adj_assign.shape[1]

    zi_all = None  # n x (K x d)
    for k in range(len(Z_i_list)):  # every cluster
        Z_i_k = Z_i_list[k]
        zi_all = Z_i_k if zi_all is None else torch.cat([zi_all, Z_i_k], dim=1)

    zi_batch = zi_all[orin_index]  # batch size x (K x d)

    # y0
    y_true_0 = torch.zeros((batch_size,1))
    if args.cuda:
        y_true_0 = y_true_0.to(device)

    if params['type'] == '0':
        W1 = params['W1'][0][0]
        W1 = torch.FloatTensor(W1)
        if args.cuda:
            W1 = W1.to(device)

        y_true = torch.empty([batch_size, num_assign, 1], dtype=torch.float, device=args.device)  # batch size x R x 1
        for j in range(adj_assign.shape[0]):  # each assignment
            adj_assign_j = adj_assign[j]  # size = m
            adj_assign_j = adj_assign_j.unsqueeze(0)  # 1 x m
            adj_assign_j = adj_assign_j.expand(batch_size, m)  # batch size x m

            y_true_j = torch.diag(torch.matmul(torch.matmul(torch.matmul(adj_assign_j, Zt), W1), zi_batch.T)).reshape(-1,1)
            y_true[:, j, :] = y_true_j

    elif params['type'] == '1':
        W1 = params['W1'][0][0]
        W1 = torch.FloatTensor(W1)
        W2 = params['W2'][0][0]
        W2 = torch.FloatTensor(W2)
        C1 = params['C1'][0][0][0][0]
        C = params['C'][0][0][0][0]
        if args.cuda:
            W1 = W1.to(device)
            W2 = W2.to(device)

        y_true = torch.empty([batch_size, num_assign, 1], dtype=torch.float, device=args.device)  # batch size x R x 1
        for j in range(adj_assign.shape[0]):  # each assignment
            adj_assign_j = adj_assign[j]
            adj_assign_j = adj_assign_j.unsqueeze(0)
            adj_assign_j = adj_assign_j.expand(batch_size, m)

            y_true_j = C * (C1 * torch.diag(torch.matmul(torch.matmul(torch.matmul(adj_assign_j, Zt), W1), zi_batch.T)).reshape(
                -1, 1) + torch.matmul(zi_batch, W2))

            y_true[:, j, :] = y_true_j

    return y_true, y_true_0

def loadFromFile(path):
    data = scio.loadmat(path)
    Z_i_list = data['Z_i_list']
    Zt = data['Zt']
    adj = data['adj']
    y = data['y'][0]
    C = data['C'][0]

    idx_trn_list = data['trn_idx_list']
    idx_val_list = data['val_idx_list']
    idx_tst_list = data['tst_idx_list']

    params = data['params']
    return Z_i_list, Zt, adj, y, C, idx_trn_list, idx_val_list, idx_tst_list, params


def load_data(dataset):
    if dataset == 'synthetic':
        Z_i_list, Zt, adj, YF, C, idx_trn_list, idx_val_list, idx_tst_list, params = loadFromFile('../../dataset/synthetic/synthetic_final.mat')
    elif dataset == 'amazon':
        Z_i_list, Zt, adj, YF, C, idx_trn_list, idx_val_list, idx_tst_list, params = loadFromFile(
            '../../dataset/amazon/amazon_3C.mat')
    elif dataset == 'amazon-6c':
        Z_i_list, Zt, adj, YF, C, idx_trn_list, idx_val_list, idx_tst_list, params = loadFromFile(
            '../../dataset/amazon/amazon_6C.mat')

    print("True C: ", C)
    cluster_size = [(C == i).sum() for i in range(args.K)]
    print('cluster size: ', cluster_size)

    return adj, Z_i_list, Zt, YF, idx_trn_list, idx_val_list, idx_tst_list, params, C



def experiment_ite(args):
    adj, Z_i_list, Zt, YF, idx_trn_list, idx_val_list, idx_tst_list, params, C = load_data(args.dataset)

    results_all = {'pehe': [], 'ate': []}

    for i_exp in range(0, 3):  # 10 runs of experiments
        print("============== Experiment ", str(i_exp), " =========================")
        trn_idx = idx_trn_list[i_exp]
        val_idx = idx_val_list[i_exp]
        tst_idx = idx_tst_list[i_exp]

        trn_loader, val_loader, tst_loader = Sythetic_loader(args.batch_size, adj, YF, trn_idx, val_idx, tst_idx)

        n = adj.shape[0]
        m = adj.shape[1]

        treated_rate = adj.sum() / adj.size
        print('data: ', args.dataset, ' n=', n, ' m=', m, ' K=', args.K, 'treated rate: ', treated_rate)

        #adj_assign = torch.eye(m)
        num_R = m
        adj_assign = np.random.binomial(1, 0.5, (num_R, m))  # R x m
        adj_assign = torch.FloatTensor(adj_assign)

        size_trn = len(trn_idx)
        input_treat_trn = adj[trn_idx].T
        input_treat_trn = torch.FloatTensor(input_treat_trn)

        Z_i_list = [torch.FloatTensor(zi) for zi in Z_i_list]
        Zt = torch.FloatTensor(Zt)

        dim_t = size_trn
        dim_x = m

        model = MTvae(args, dim_t, dim_x)

        # cuda
        if args.cuda:
            model = model.to(device)
            input_treat_trn = input_treat_trn.to(device)
            adj_assign = adj_assign.to(device)
            Z_i_list = [zi.to(device) for zi in Z_i_list]
            Zt = Zt.to(device)

        par_t = list(model.mu_zt.parameters()) + list(model.logvar_zt.parameters()) + list(
            model.a_reconstby_zt.parameters())
        par_z = list(model.mu_zi_k.parameters()) + list(model.logvar_zi_k.parameters())
        par_y = list(model.y_pred_1.parameters()) + list(model.y_pred_2.parameters()) + par_z
        optimizer_1 = optim.Adam([{'params': par_t, 'lr': args.lr}], weight_decay=args.weight_decay)  # zt
        optimizer_2 = optim.Adam([{'params': [model.mu_p_zt, model.logvar_p_zt], 'lr': 0.01}],
                                 weight_decay=args.weight_decay)  # centroid
        optimizer_3 = optim.Adam([{'params': par_z, 'lr': args.lr}], weight_decay=args.weight_decay)  # zi
        optimizer_4 = optim.Adam([{'params': par_y, 'lr': args.lr}], weight_decay=args.weight_decay)  # y
        optimizer = [optimizer_1, optimizer_2, optimizer_3, optimizer_4]
        train(args.epochs, model, trn_loader, val_loader, tst_loader, input_treat_trn, adj_assign, Z_i_list, Zt, params,
              C, optimizer)
        eval_result_tst = test(model, tst_loader, input_treat_trn, adj_assign, Z_i_list, Zt, params, C,
                               show_cluster=True)

        results_all['pehe'].append(eval_result_tst['pehe'])
        results_all['ate'].append(eval_result_tst['ate'])

        break  # remove if more experiments are needed

    results_all['average_pehe'] = np.mean(np.array(results_all['pehe'], dtype=np.float))
    results_all['std_pehe'] = np.std(np.array(results_all['pehe'], dtype=np.float))
    results_all['average_ate'] = np.mean(np.array(results_all['ate'], dtype=np.float))
    results_all['std_ate'] = np.std(np.array(results_all['ate'], dtype=np.float))

    print("============== Overall experiment results =========================")
    for k in results_all:
        if isinstance(results_all[k], list):
            print(k, ": ", results_all[k])
        else:
            print(k, f": {results_all[k]:.4f}")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))


if __name__ == '__main__':
    t_begin = time.time()

    if args.dataset == 'synthetic':
        args.K = 4
    elif args.dataset == 'amazon':
        args.K = 3
    elif args.dataset == 'amazon-6c':
        args.K = 6
    experiment_ite(args)

