import os

import numpy as np

from src.util.array_util import data_to_sparse
from src.util.io import load_pickle, save_pickle

model_type = 'global_model'


def train_global_model(train, val, test, results_dir, dataset_name, overwrite=False, save_multinomials=True,
                       is_eval=True):
    """
    Runs the Global experiment of the paper.

    Data come in COO form. That is a numpy array of (N x 3) where each row is the (row, column, value) triplet of the
    sparse array UID x PID. N is the number of entries in the array.

    :param train: Train data COO matrix.
    :param val: Validation data COO matrix.
    :param test: Test data COO matrix.
    :param results_dir: Name of the directory the results will be saved.
    :param dataset_name: Name of the directory the results will be saved.
    :param overwrite: Boolean, on whether to overwrite the multinomials or read them if they exist.
    :param save_multinomials: Boolean, on whether to save the multinomials.
    :param is_eval: Boolean, that defines if this data will be used for evaluation or training. The difference is that
    when evaluating, the validation data is added to the training, otherwise it is not.
    :return: return multinomials, which is a dense matrix of predicted probability
    """
    filename = os.path.join(results_dir, model_type, dataset_name, 'user_multinomials.pkl')
    if os.path.exists(filename) and not overwrite:
        all_multinomials = load_pickle(filename)
    else:
        if is_eval:
            train = np.vstack((train, val))
        train_matrix = data_to_sparse(train).tocsr()
        all_multinomials = construct_multinomials(train_matrix, test)
        if save_multinomials:
            save_pickle(filename, all_multinomials, False)
    return all_multinomials


def construct_multinomials(train_matrix, test):
    shape = data_to_sparse(test).shape
    item_weights = train_matrix.sum(axis=0)
    result = np.repeat(item_weights, shape[0], axis=0)
    result = np.asarray(result, dtype=np.float32)
    return result
