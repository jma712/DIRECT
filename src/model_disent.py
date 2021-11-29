# change to the latest version 2020.10.25

import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda:0")

class MTvae(nn.Module):
    def __init__(self, args, dim_t, dim_x):
        super(MTvae, self).__init__()

        self.device = args.device

        dim_input_t = dim_t
        dim_input_i = dim_x
        self.num_cluster = args.K
        self.nogb = args.nogb
        self.dim_zt = args.dim_zt
        self.dim_zi = args.dim_zi

        # Recognition model
        # q(Z^T|A)
        self.logvar_zt = nn.Sequential(nn.Linear(dim_input_t, args.dim_zt))  #nn.Tanh()
        self.mu_zt = nn.Sequential(nn.Linear(dim_input_t, args.dim_zt))
        # q(C|Z^T)
        self.qc = nn.Sequential(nn.Linear(args.dim_zt, args.dim_zt), nn.ReLU(), nn.Linear(args.dim_zt, args.K))
        # p(a|z^T)
        self.a_reconstby_zt = nn.Sequential(nn.Linear(args.dim_zt, dim_input_t))
        # q(Z^(I,K)|A,C)
        self.mu_zi_k = nn.Linear(dim_input_i, args.dim_zi)
        self.logvar_zi_k = nn.Linear(dim_input_i, args.dim_zi)

        # prior generator
        # p(Z^T|C)
        self.mu_p_zt = torch.normal(mean=0, std=1, size=(args.K, args.dim_zt), device=args.device)
        self.mu_p_zt = torch.nn.Parameter(self.mu_p_zt, requires_grad=True)
        self.register_parameter("mu_p_zt", self.mu_p_zt)
        self.mu_p_wt = args.mu_p_wt
        self.logvar_p_zt = torch.nn.Parameter(torch.ones((args.K, args.dim_zt),device=args.device), requires_grad=True)
        self.register_parameter("logvar_p_zt", self.logvar_p_zt)

        # Generative model
        # predict outcome
        self.y_pred_1 = nn.Linear(self.dim_zt, self.num_cluster * self.dim_zi)
        self.y_pred_2 = nn.Linear(self.num_cluster * self.dim_zi, 1)
        self.logvar_y = nn.Linear(self.dim_zt, self.num_cluster * self.dim_zi)

    def encode_t(self, input_treat):
        mu_zt = self.mu_zt(input_treat)
        logvar_zt = self.logvar_zt(input_treat)

        return mu_zt, logvar_zt

    def get_treat_rep(self, input_treat_new):
        # encoder: zt, zi
        mu_zt, logvar_zt = self.encode_t(input_treat_new)
        zt_sample = self.reparameterize(mu_zt, logvar_zt)  # m x d
        qc = self.compute_qc(zt_sample)  # m x k, unnormalized logits
        cates = F.softmax(qc, dim=1)  # normalize with softmax, m x k

        return mu_zt, logvar_zt, cates

    def compute_qc(self, zt_sample, type='cos'):
        # zt_sample: m x d_t
        if type == 'cos':
            cos = nn.CosineSimilarity(dim=2, eps=1e-6)
            zt_sample_K = torch.unsqueeze(zt_sample, dim=0).repeat(self.num_cluster, 1, 1)  # K x m x d_t
            mu_p_zt = torch.unsqueeze(self.mu_p_zt, dim=1).repeat(1, zt_sample_K.shape[1], 1)
            cos_similarity = cos(self.mu_p_wt * mu_p_zt, zt_sample_K)
            qc = cos_similarity.T
            qc *= 10
        elif type == 'linear':
            qc = self.qc(zt_sample)
        elif type == 'euc':

            zt_sample_K = torch.unsqueeze(zt_sample, dim=0).repeat(self.num_cluster, 1, 1)  # K x m x d_t
            mu_p_zt = torch.unsqueeze(self.mu_p_zt, dim=1).repeat(1, zt_sample_K.shape[1], 1)

            distance = torch.norm(zt_sample_K - self.mu_p_wt * mu_p_zt, dim=-1)  # k x m
            qc = 1.0/(distance.T + 1)
        return qc

    def encode_i(self, input_ins, qc):
        mu_zi_list = []
        logvar_zi_list = []

        # c sample
        if self.training:
            cates = nn.functional.gumbel_softmax(logits=qc, hard=True)  # one-hot, m x k
        else:
            cates = F.softmax(qc, dim=1)  # normalize with softmax, m x k

        # q(z^(I,k)|a^k)
        for k in range(self.num_cluster):
            cates_k = cates[:, k].reshape(1, -1)  # 1 x m, cates_k[j]=1: item j is in cluster k

            # q-network
            x_k = input_ins * cates_k
            mu_k = self.mu_zi_k(x_k)
            logvar_k = self.logvar_zi_k(x_k)

            mu_zi_list.append(mu_k)
            logvar_zi_list.append(logvar_k)

        return mu_zi_list, logvar_zi_list, cates

    def reparameterize(self, mu, logvar):
        '''
        z = mu + std * epsilon
        '''
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    # p(A|Z^T,Z^I,C)
    def decoder(self, zt_sample, zi_sample_list, c_sample):
        probs = None
        for k in range(self.num_cluster):
            cates_k = c_sample[:, k].reshape(1, -1)

            zi_sample_k = zi_sample_list[k]  # n x d
            a_pred_k = torch.matmul(zi_sample_k, zt_sample.T)  # n x m
            a_pred_k = torch.sigmoid(a_pred_k)
            a_pred_k = a_pred_k * cates_k
            probs = (a_pred_k if (probs is None) else (probs + a_pred_k))

        return probs

    def predictY(self, zt_sample, zi_sample_list, c_sample, adj_batch):
        # concat zi
        zi_all = None  # batch x (K x d)
        for k in range(len(zi_sample_list)):  # every cluster
            Z_i_k = zi_sample_list[k]
            zi_all = Z_i_k if zi_all is None else torch.cat([zi_all, Z_i_k], dim=1)

        a_zt = torch.matmul(adj_batch, zt_sample)  # batch size x d_t

        rep_w1 = self.y_pred_1(a_zt)
        pred_y2 = self.y_pred_2(zi_all)
        mu_y = torch.matmul(rep_w1, zi_all.T).diag().view(-1, 1) + pred_y2

        logvar_y = torch.ones_like(mu_y).to(device)

        return mu_y, logvar_y

    def forward(self, input_ins, input_treat):
        # encoder: zt, zi
        mu_zt, logvar_zt = self.encode_t(input_treat)
        zt_sample = self.reparameterize(mu_zt, logvar_zt)  # sample zt: m x d
        qc = self.compute_qc(zt_sample)  # m x k, unnormalized logits
        qc = F.softmax(qc, dim=1)  # normalize with softmax, m x k

        cates = qc

        mu_zi_list, logvar_zi_list, c_sample = self.encode_i(input_ins, qc)

        a_reconstby_zt = self.a_reconstby_zt(zt_sample)
        a_reconstby_zt = torch.sigmoid(a_reconstby_zt)  # m x n

        # sample zi
        zi_sample_list = []  # size = k, each elem is n x d
        for k in range(self.num_cluster):
            mu_zi_k = mu_zi_list[k]
            logvar_zi_k = logvar_zi_list[k]
            zi_sample_k = self.reparameterize(mu_zi_k, logvar_zi_k)
            zi_sample_list.append(zi_sample_k)

        # decoder
        a_pred = self.decoder(zt_sample, zi_sample_list, c_sample)
        mu_y, logvar_y = self.predictY(zt_sample, zi_sample_list, c_sample, input_ins)  # n x 1

        return mu_zt, logvar_zt, self.mu_p_zt, self.logvar_p_zt, cates, mu_zi_list, logvar_zi_list, zi_sample_list, a_pred, mu_y, logvar_y, a_reconstby_zt
