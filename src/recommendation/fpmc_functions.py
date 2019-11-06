"""From https://github.com/khesui/FPMC. """

import math

import numpy as np
from numba import jit


class FPMC(object):
    def __init__(self, n_user, n_item, n_factor, learn_rate, regular):
        self.user_set = set()
        self.item_set = set()

        self.n_user = n_user
        self.n_item = n_item

        self.n_factor = n_factor
        self.learn_rate = learn_rate
        self.regular = regular

    # noinspection PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit
    def init_model(self, std=0.01):
        self.VUI = np.random.normal(0, std, size=(self.n_user, self.n_factor))
        self.VIU = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VIL = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VLI = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VUI_m_VIU = np.dot(self.VUI, self.VIU.T)
        self.VIL_m_VLI = np.dot(self.VIL, self.VLI.T)

    def learnSBPR_FPMC_multinomials(self, tr_data, te_data, n_epoch=2, neg_batch_size=10):
        tr_3_list = data_to_3_list(tr_data)
        for epoch in range(n_epoch):
            self.learn_epoch(tr_3_list, neg_batch_size)
        user_multinomials = self.construct_multinomials(te_data)
        return user_multinomials

    # noinspection PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit
    def learn_epoch(self, data_3_list, neg_batch_size):
        VUI, VIU, VLI, VIL = learn_epoch_jit(data_3_list[0], data_3_list[1], data_3_list[2], neg_batch_size,
                                             np.array(list(self.item_set)), self.VUI, self.VIU, self.VLI, self.VIL,
                                             self.learn_rate, self.regular)
        self.VUI = VUI
        self.VIU = VIU
        self.VLI = VLI
        self.VIL = VIL

    def construct_multinomials(self, test_data_list):
        np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)

        shape = (self.n_user, self.n_item)
        result = np.zeros(shape, dtype=np.float32)

        for (u, b_tm1) in test_data_list:
            b_tm1 = list(b_tm1)
            scores = compute_x_batch_jit(u, b_tm1, self.VUI_m_VIU, self.VIL_m_VLI)
            predictions = l1_norm(scores)
            result[u] += predictions
        return result


@jit(nopython=True)
def compute_x_jit(u, i, b_tm1, VUI, VIU, VLI, VIL):
    acc_val = 0.0
    for l in b_tm1:
        acc_val += np.dot(VIL[i], VLI[l])
    return np.dot(VUI[u], VIU[i]) + (acc_val / len(b_tm1))


@jit(nopython=True)
def learn_epoch_jit(u_list, i_list, b_tm1_list, neg_batch_size, item_set, VUI, VIU, VLI, VIL, learn_rate, regular):
    for iter_idx in range(len(u_list)):
        d_idx = np.random.randint(0, len(u_list))
        u = u_list[d_idx]
        i = i_list[d_idx]
        b_tm1 = b_tm1_list[d_idx][b_tm1_list[d_idx] != -1]

        j_list = np.random.choice(item_set, size=neg_batch_size, replace=False)

        z1 = compute_x_jit(u, i, b_tm1, VUI, VIU, VLI, VIL)
        for j in j_list:
            z2 = compute_x_jit(u, j, b_tm1, VUI, VIU, VLI, VIL)
            delta = 1 - sigmoid_jit(z1 - z2)

            VUI_update = learn_rate * (delta * (VIU[i] - VIU[j]) - regular * VUI[u])
            VIUi_update = learn_rate * (delta * VUI[u] - regular * VIU[i])
            VIUj_update = learn_rate * (-delta * VUI[u] - regular * VIU[j])

            VUI[u] += VUI_update
            VIU[i] += VIUi_update
            VIU[j] += VIUj_update

            eta = np.zeros(VLI.shape[1])
            for l in b_tm1:
                eta += VLI[l]
            eta = eta / len(b_tm1)

            VILi_update = learn_rate * (delta * eta - regular * VIL[i])
            VILj_update = learn_rate * (-delta * eta - regular * VIL[j])
            VLI_updates = np.zeros((len(b_tm1), VLI.shape[1]))
            for idx, l in enumerate(b_tm1):
                VLI_updates[idx] = learn_rate * ((delta * (VIL[i] - VIL[j]) / len(b_tm1)) - regular * VLI[l])

            VIL[i] += VILi_update
            VIL[j] += VILj_update
            for idx, l in enumerate(b_tm1):
                VLI[l] += VLI_updates[idx]

    return VUI, VIU, VLI, VIL


@jit(nopython=True)
def sigmoid_jit(x):
    if x >= 0:
        return math.exp(-np.logaddexp(0, -x))
    else:
        return math.exp(x - np.logaddexp(x, 0))


@jit(nopython=True)
def compute_x_batch_jit(u, b_tm1, VUI_m_VIU, VIL_m_VLI):
    former = VUI_m_VIU[u]
    latter = np.zeros(VIL_m_VLI.shape[0])
    for idx in range(VIL_m_VLI.shape[0]):
        for l in b_tm1:
            latter[idx] += VIL_m_VLI[idx, l]
    latter = latter / len(b_tm1)
    return former + latter


def l1_norm(arr):
    # convert to non-negative scores that sum to 1
    arr1 = arr - arr.min()
    arr2 = arr1 / np.linalg.norm(arr1, ord=1)
    return arr2


def data_to_3_list(data_list):
    u_list = []
    i_list = []
    b_tm1_list = []
    max_l = 0
    for d in data_list:
        u_list.append(d[0])
        i_list.append(d[1])
        b_tm1_list.append(list(d[2]))
        if len(d[2]) > max_l:
            max_l = len(d[2])
    for b_tm1 in b_tm1_list:
        b_tm1.extend([-1 for i in range(max_l - len(b_tm1))])
    b_tm1_list = np.array(b_tm1_list)
    return u_list, i_list, b_tm1_list
