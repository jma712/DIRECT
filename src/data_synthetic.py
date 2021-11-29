'''
generate a synthetic dataset
2020.7
'''
import numpy as np
import random
from scipy.sparse import csc_matrix
import math
import scipy.io as scio

import json
import gzip
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as tsn
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

#from nltk import word_tokenize
#from nltk.stem import PorterStemmer, WordNetLemmatizer

random.seed(1)
np.random.seed(1)
saving = False

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def row_norm(a):
    # for normalize the AZ matrix, which can be too huge
    row_sums = a.sum(axis=1)
    row_sums[np.where(row_sums == 0)] = 1  # if sum = 0
    norm_a = a.astype(float) / row_sums[:, np.newaxis]
    return norm_a

def generate_y_final(Z_i_list, Zt, adj, params=None, mode="1"):
    #
    K = len(Z_i_list)
    zi_all = None  # n x (K x d)
    for k in range(len(Z_i_list)):  # every cluster
        Z_i_k = Z_i_list[k]
        zi_all = Z_i_k if zi_all is None else np.concatenate([zi_all, Z_i_k], axis=1)

    d_x = zi_all.shape[1]  # (K x d)
    d_t = Zt.shape[1]  # d_t

    n = zi_all.shape[0]
    m = Zt.shape[0]

    if mode == "0":
        if params == None:
            W1 = np.random.normal(0.1, 0.01, size=(d_t, d_x))
            params = {'W1': W1, 'type': mode}
        else:
            W1 = params['W1']

        y = np.diag(np.matmul(np.matmul(np.matmul(adj, Zt), W1), zi_all.T))
        y = y.reshape(-1)

    if mode == "1":
        if params == None:
            W1 = np.random.normal(0.1, 0.01, size=(d_t, d_x))
            W2 = np.random.normal(100, 0.0001, size=(d_x, 1))
            C1 = 0.1
            C = 0.012
            params = {'W1': W1, 'type': mode, 'W2': W2, 'C1': C1, 'C': C}
        else:
            W1 = params['W1']
            W2 = params['W2']
            C1 = params['C1']
            C = params['C']

        y = C * (C1 * np.diag(np.matmul(np.matmul(np.matmul(adj, Zt), W1), zi_all.T)).reshape(-1) + np.dot(zi_all, W2).reshape(-1))
        y1 = np.diag(np.matmul(np.matmul(np.matmul(adj, Zt), W1), zi_all.T)).reshape(-1)
        y2 = np.dot(zi_all, W2).reshape(-1)

    # statistics
    print('generate y with type: ', mode, '; observed y mean/std: ', np.mean(y), np.std(y), ' y1:', np.mean(y1), np.std(y1),
          ' y2:', np.mean(y2), np.std(y2))
    return y, params

def get_y_final(Z_i_list, Zt, adj_assign, params):
    zi_all = None  # n x (K x d)
    for k in range(len(Z_i_list)):  # every cluster
        Z_i_k = Z_i_list[k]
        zi_all = Z_i_k if zi_all is None else np.concatenate([zi_all, Z_i_k], axis=1)

    n = Z_i_list[0].shape[0]
    m = Zt.shape[0]

    if params['type'] == "0":
        y0_true = np.zeros(n)
        y_true_list = []
        for j in range(adj_assign.shape[0]):  # each assignment
            adj_assign_j = adj_assign[j]  # size = m
            adj_assign_j = np.tile(adj_assign_j, (n, 1))  # n x m

            y, _ = generate_y_final(Z_i_list, Zt, adj_assign_j, params=params)  # n

            y_true_list.append(y)

    elif params['type'] == "1":
        y0_true = np.zeros(n)
        y_true_list = []
        for j in range(adj_assign.shape[0]):  # each assignment
            adj_assign_j = adj_assign[j]  # size = m
            adj_assign_j = np.tile(adj_assign_j, (n, 1))  # n x m

            y, _ = generate_y_final(Z_i_list, Zt, adj_assign_j, params=params)  # n

            y_true_list.append(y)

    return y_true_list, y0_true


def plot_cluster(Zt, C, num_cluster, mu_zt_all=None, saving=False, Zt_tsn=None, title=None):
    cluster_color = ['red', 'blue', 'green', 'black', 'yellow', 'purple']

    # print("centroid: ", mu_zt_all)
    fig, ax = plt.subplots()
    if Zt_tsn is None:
        if mu_zt_all is not None:
            Zt_and_center = np.concatenate((Zt, mu_zt_all), axis=0)  # (m + K) x d
        else:
            Zt_and_center = Zt
        Zt_tsn = tsn(n_components=2).fit_transform(Zt_and_center)  # m x d => m x 2
    for k in range(num_cluster):
        idx_k = np.where(C == k)
        if len(idx_k[0]) > 0:
            ax.scatter(Zt_tsn[idx_k, 0], Zt_tsn[idx_k, 1], 3, marker='o', color=cluster_color[k])  # cluster k
            if mu_zt_all is not None:
                ax.scatter(Zt_tsn[k - num_cluster, 0], Zt_tsn[k - num_cluster, 1], 100, marker='D',
                       color=cluster_color[k])  # centroid k

        # plt.xlim(-100, 100)
    if not title is None:
        plt.title(title)

    if saving:
        plt.savefig('./figs/sythetic_data_tsne.pdf', bbox_inches='tight')
    else:
        plt.show()
    return Zt_tsn


def data_generate(n, d_x, m, d_t, num_cluster):
    Z_i_list = []
    for k in range(num_cluster):
        Z_i_k = np.random.normal(0, 1, (n, d_x))
        Z_i_list.append(Z_i_k)
    Z_i_list = np.array(Z_i_list)

    # mu_k, k x d_t
    mu_zt_all = np.random.normal(0, 0.4, (num_cluster, d_t))
    std_zt_all = np.random.random((num_cluster, d_t))

    # C, m x k
    pi = np.random.random(num_cluster)
    pi = pi / pi.sum()  # normailize, sum to 1

    # Z^t
    zt_list = []
    size_cur = 0
    C = np.zeros(m)

    for k in range(num_cluster):
        mu_zt_k = mu_zt_all[k]
        std_zt_k = std_zt_all[k]
        if k == num_cluster - 1:
            size_k = m - size_cur
        else:
            size_k = int(m * pi[k])
        C[size_cur:size_cur+size_k] = k
        zt_k_sample = np.random.multivariate_normal(mu_zt_k, np.diag(std_zt_k), size_k)  # size_k x d
        size_cur += size_k

        zt_list.append(zt_k_sample)

    Zt = np.concatenate(zt_list, axis=0)  # m x d

    # treatment assignment
    adj = np.zeros((n, m))
    for k in range(num_cluster):
        zi_k = Z_i_list[k]  # n x d
        c_k = (C == k)  # m x 1, binary 0/1
        c_k = c_k.reshape((-1, 1))
        zt_k = c_k * Zt
        adj_k = np.matmul(zi_k, zt_k.T)  # n x m
        adj = adj + adj_k

    adj = sigmoid(adj)
    adj[np.where(adj > 0.5)] = 1.0
    adj[np.where(adj <= 0.5)] = 0.0

    # y
    y, params = generate_y_final(Z_i_list, Zt, adj, params=None)

    #plot_cluster(Zt, C, num_cluster, mu_zt_all=mu_zt_all, saving=False)
    #plot_cluster(adj.T, C, num_cluster)

    # statistics report
    print("C: ", C)
    print("treated rate: ", adj.sum()/adj.size)
    # adj_single = np.eye(m)  # numpy, r x m
    # y_true_list, y0_true = get_y_final(Z_i_list, Zt, adj_single, params)
    #
    # single_ate = np.zeros(len(adj_single))
    # for r in range(len(adj_single)): # R
    #     ate_r = np.mean(y_true_list[r] - y0_true)  # n
    #     single_ate[r] = ate_r
    # ave_single_ate = single_ate.mean()
    # print("average single cause ate: ", ave_single_ate)

    return Z_i_list, Zt, adj, y, C, params


def get_y_true_new(Z_i_list, Zt, adj_assign, W1, W2):
    zi_all = None  # n x (K x d)
    for k in range(len(Z_i_list)):  # every cluster
        Z_i_k = Z_i_list[k]
        zi_all = Z_i_k if zi_all is None else np.concatenate([zi_all, Z_i_k], axis=1)

    n = Z_i_list[0].shape[0]
    m = Zt.shape[0]

    y0_true = np.zeros(n)

    y_true_list = []
    for j in range(adj_assign.shape[0]):  # each assignment
        adj_assign_j = adj_assign[j]  # size = m
        adj_assign_j = np.tile(adj_assign_j, (n, 1))  # n x m

        y = np.diag(np.matmul(np.matmul(zi_all, W1), adj_assign_j.T)).reshape(-1,1) + np.matmul(np.matmul(adj_assign_j, Zt), W2)
        y = y.reshape(-1)  # n

        y_true_list.append(y)

    return y_true_list, y0_true


def generate_treat(vocal, words_per_treat, num_treat):
    treatments = []
    for i in range(num_treat):
        sample = set(random.sample(vocal, words_per_treat))
        if sample not in treatments:
            treatments.append(sample)
    return treatments

def amazon_data_prep2():
    # instrument
    vocal_1 = ['sound', 'pedal', 'string', 'record', 'rock', 'musician', 'ear', 'classic', 'loud', 'tuner', 'amp',
               'acoustic', 'capo', 'tune', 'tuning', 'recording', 'microphone',  'speaker', 'player']
    # cloth
    vocal_2 = ['fit', 'size', 'wear', 'comfortable', 'loose', 'fitting','sizing', 'longer', 'large', 'small',
               'material', 'soft', 'comfort', 'wore', 'lightweight','stylish']

    vocal_2_norepeat = [v for v in vocal_2 if v not in vocal_1]
    vocal = vocal_1 + vocal_2_norepeat

    # electronic and accessories
    vocal_3 = ['phone', 'screen', 'case', 'charge', 'device', 'battery', 'charger', 'cover', 'protection', 'headphone',
               'long', 'quick', 'access', 'connect']

    vocal_3_norepeat = [v for v in vocal_3 if v not in vocal]
    vocal = vocal + vocal_3_norepeat

    # randomly generate word sequences
    words_per_treat = 3  # the number of words in each treatment
    num_treat = 34

    treat_vol1 = generate_treat(vocal_1, words_per_treat, num_treat+7)
    treat_vol2 = generate_treat(vocal_2, words_per_treat, num_treat)
    treat_vol3 = generate_treat(vocal_3, words_per_treat, num_treat)

    treat2_norepeat = [t for t in treat_vol2 if t not in treat_vol1]
    treat_all = treat_vol1 + treat2_norepeat
    treat3_norepeat = [t for t in treat_vol3 if t not in treat_all]
    treat_all = treat_all + treat3_norepeat

    print('num of treatments: ', len(treat_all))

    path_list = ['/Users/heather/Downloads/Musical_Instruments_5.json',
                 '/Users/heather/Downloads/Clothing_Shoes_and_Jewelry_5.json',
                 '/Users/heather/Downloads/Cell_Phones_and_Accessories_5.json']

    corpus = []
    path_i = 0
    for path in path_list:
        item_reviewNum = {}
        item_review = {}
        with open(path) as f:
            for line in f:
                data_line = json.loads(line)
                asin = data_line['asin']  # item ID
                reviewText = data_line['reviewText']
                item_reviewNum[asin] = 0 if asin not in item_reviewNum else item_reviewNum[asin] + 1
                item_review[asin] = reviewText if asin not in item_review else item_review[asin] + ' ' + reviewText

        sorted_item_reviewNum = sorted(item_reviewNum.items(), key=lambda kv: kv[1], reverse=True)
        print('number of items in current category in total: ', len(sorted_item_reviewNum))

        num_top_items = 1000  # items receive most number of reviews
        if path_i == 1:
            num_top_items = 1100
        print("most number of reviews in current category: ", sorted_item_reviewNum[:num_top_items])

        selected_top_items = [i[0] for i in sorted_item_reviewNum[:num_top_items]]

        # dictionary for the top-item reviews
        corpus_cur = [item_review[i] for i in selected_top_items]
        corpus = corpus + corpus_cur  # selected instance x string
        path_i += 1

    cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=LemmaTokenizer(),
                         vocabulary=vocal)
    cv_fit = cv.fit_transform(corpus)

    # sequence of words
    cv_array = cv_fit.toarray()
    treat_assign = np.zeros((len(cv_array), len(treat_all)))  # n x m
    for i in range(len(cv_array)):  # each instance
        for ti in range(len(treat_all)):
            flag = True  # a(i,ti)=1
            treat_i = treat_all[ti]  # a set of word
            word_index = [vocal.index(word) for word in treat_i]
            for wi in word_index:
                if cv_array[i][wi] < 1:
                    flag = False
                    break
            if flag:
                treat_assign[i][ti] = 1

    treat_all = [list(t) for t in treat_all]
    return treat_assign, treat_all

def amazon_data_prep_6c():
    # instrument
    vocal_1 = ['sound', 'pedal', 'string', 'record', 'rock', 'musician', 'ear', 'classic', 'loud', 'tuner', 'amp',
               'acoustic', 'capo', 'tune', 'tuning', 'recording', 'microphone',  'speaker', 'player']
    # cloth
    vocal_2 = ['fit', 'size', 'wear', 'comfortable', 'loose', 'fitting','sizing', 'longer', 'large', 'small',
               'material', 'soft', 'comfort', 'wore', 'lightweight','stylish']

    vocal_2_norepeat = [v for v in vocal_2 if v not in vocal_1]
    vocal = vocal_1 + vocal_2_norepeat

    # electronic and accessories
    vocal_3 = ['phone', 'screen', 'case', 'charge', 'device', 'battery', 'charger', 'cover', 'protection', 'headphone',
               'long', 'quick', 'access', 'connect']
    vocal_3_norepeat = [v for v in vocal_3 if v not in vocal]
    vocal = vocal + vocal_3_norepeat

    # Sports_and_Outdoors
    vocal_4 = ['workout', 'exercise', 'exercises', 'carry', 'weight', 'safe', 'convenient', 'inconvenient', 'safety', 'gun',
               'weapon', 'travel', 'traveling', 'camping', 'shot']
    vocal_4_norepeat = [v for v in vocal_4 if v not in vocal]
    vocal = vocal + vocal_4_norepeat

    # Beauty
    vocal_5 = ['smell', 'beauty', 'face', 'eye', 'skin', 'smelling', 'clean', 'cleansing', 'oil', 'oily', 'wet', 'dry', 'color',
               'makeup', 'night', 'wash', 'greasy', 'moisturizer', 'sensitive']
    vocal_5_norepeat = [v for v in vocal_5 if v not in vocal]
    vocal = vocal + vocal_5_norepeat

    # Baby
    vocal_6 = ['month', 'year', 'old', 'baby', 'feeding', 'feed', 'hospital', 'nursing', 'daughter', 'son', 'eat', 'eating',
               'drink', 'sleep', 'play', 'education', 'toy', 'fun', 'game', 'diaper', 'poop', 'teether', 'teething']
    vocal_6_norepeat = [v for v in vocal_6 if v not in vocal]
    vocal = vocal + vocal_6_norepeat

    # randomly generate word sequences
    words_per_treat = 3  # the number of words in each treatment
    num_treat = 55

    treat_vol1 = generate_treat(vocal_1, words_per_treat, num_treat+7)
    treat_vol2 = generate_treat(vocal_2, words_per_treat, num_treat)
    treat_vol3 = generate_treat(vocal_3, words_per_treat, num_treat)
    treat_vol4 = generate_treat(vocal_4, words_per_treat, num_treat)
    treat_vol5 = generate_treat(vocal_5, words_per_treat, num_treat)
    treat_vol6 = generate_treat(vocal_6, words_per_treat, num_treat)

    treat2_norepeat = [t for t in treat_vol2 if t not in treat_vol1]
    treat_all = treat_vol1 + treat2_norepeat
    treat3_norepeat = [t for t in treat_vol3 if t not in treat_all]
    treat_all = treat_all + treat3_norepeat
    treat4_norepeat = [t for t in treat_vol4 if t not in treat_all]
    treat_all = treat_all + treat4_norepeat
    treat5_norepeat = [t for t in treat_vol5 if t not in treat_all]
    treat_all = treat_all + treat5_norepeat
    treat6_norepeat = [t for t in treat_vol6 if t not in treat_all]
    treat_all = treat_all + treat6_norepeat

    print('num of treatments: ', len(treat_all))

    path_list = ['/Users/heather/Downloads/Musical_Instruments_5.json',
                 '/Users/heather/Downloads/Clothing_Shoes_and_Jewelry_5.json',
                 '/Users/heather/Downloads/Cell_Phones_and_Accessories_5.json',
                 '/Users/heather/Downloads/Sports_and_Outdoors_5.json',
                 '/Users/heather/Downloads/Beauty_5.json',
                 '/Users/heather/Downloads/Baby_5.json']

    corpus = []
    path_i = 0
    for path in path_list:
        item_reviewNum = {}
        item_review = {}
        with open(path) as f:
            for line in f:
                data_line = json.loads(line)
                asin = data_line['asin']  # item ID
                reviewText = data_line['reviewText']
                item_reviewNum[asin] = 0 if asin not in item_reviewNum else item_reviewNum[asin] + 1
                item_review[asin] = reviewText if asin not in item_review else item_review[asin] + ' ' + reviewText

        sorted_item_reviewNum = sorted(item_reviewNum.items(), key=lambda kv: kv[1], reverse=True)
        print('number of items in current category in total: ', len(sorted_item_reviewNum))

        num_top_items = 1000  # items receive most number of reviews
        if path_i == 1:
            num_top_items = 1100
        print("most number of reviews in current category: ", sorted_item_reviewNum[:num_top_items])

        selected_top_items = [i[0] for i in sorted_item_reviewNum[:num_top_items]]

        # dictionary for the top-item reviews
        corpus_cur = [item_review[i] for i in selected_top_items]
        corpus = corpus + corpus_cur  # selected instance x string

        path_i += 1

    cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=LemmaTokenizer(),
                         vocabulary=vocal)
    cv_fit = cv.fit_transform(corpus)

    # sequence of words
    cv_array = cv_fit.toarray()
    treat_assign = np.zeros((len(cv_array), len(treat_all)))  # n x m
    for i in range(len(cv_array)):  # each instance
        for ti in range(len(treat_all)):
            flag = True  # a(i,ti)=1
            treat_i = treat_all[ti]  # a set of word
            word_index = [vocal.index(word) for word in treat_i]
            for wi in word_index:
                if cv_array[i][wi] < 1:
                    flag = False
                    break
            if flag:
                treat_assign[i][ti] = 1

    treat_all = [list(t) for t in treat_all]
    return treat_assign, treat_all


if __name__ == '__main__':
    saving_in_file = True
    dataname = 'synthetic'

    if dataname == 'synthetic':
        n = 2500
        m = 500
        d_x = 20
        d_t = 20
        num_cluster = 4
        trn_rate = 0.6
        val_rate = 0.2
        tst_rate = 0.2
        size_trn = int(trn_rate * n)
        size_tst = int(tst_rate * n)
        size_val = n - size_trn - size_tst

        Z_i_list, Zt, adj, y, C, params = data_generate(n, d_x, m, d_t, num_cluster)

        idx_all = np.arange(n)
        trn_idx_list, val_idx_list, tst_idx_list = [], [], []
        for i in range(10):  # 10 experiments
            np.random.shuffle(idx_all)
            trn_id = idx_all[:size_trn]
            val_id = idx_all[size_trn: size_trn + size_val]
            tst_id = idx_all[size_trn + size_val:]
            trn_idx_list.append(trn_id.copy())
            val_idx_list.append(val_id.copy())
            tst_idx_list.append(tst_id.copy())

        adj_assign = np.eye(m)

        print('data generated!')
        if saving_in_file:
            scio.savemat('../dataset/synthetic/synthetic_final.mat', {
                'Z_i_list': Z_i_list, 'Zt': Zt,
                'adj': adj, 'y': y,
                'C': C,
                'adj_eval': adj_assign,
                'params': params,
                'trn_idx_list': trn_idx_list, 'val_idx_list': val_idx_list, 'tst_idx_list': tst_idx_list
            })
            print('data saved!')

    elif dataname == 'amazon':
        cv_fit, word_name_select = amazon_data_prep2()  # use a dictionary, randomly select some words as a treatment

        if saving_in_file:
            scio.savemat('../dataset/amazon_pre_3word.mat', {
                'cv_fit': cv_fit, 'vocalbulary': word_name_select,
            })
            print('data saved!')

    elif dataname == 'amazon-6c':
        cv_fit, word_name_select = amazon_data_prep_6c()  # use a dictionary, randomly select some words as a treatment

        if saving_in_file:
            scio.savemat('../dataset/amazon_pre_3word_6c.mat', {
                'cv_fit': cv_fit, 'vocalbulary': word_name_select,
            })
            print('data saved!')









