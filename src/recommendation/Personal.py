import os

import numpy as np

from src.recommendation.mixture_functions import get_train
from src.util.io import load_pickle, save_pickle

model_type = 'personal_model'


def train_favourite_model(train, val, test, results_dir, dataset_name, overwrite=False, save_multinomials=True,
                          is_eval=True):
    """
    Runs the Personal experiment of the paper.

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
        train_matrix = get_train(train, val, test, is_eval)
        all_multinomials = construct_multinomials_favourite(train_matrix)
        if save_multinomials:
            save_pickle(filename, all_multinomials, False)
    return all_multinomials


def construct_multinomials_favourite(train_matrix):
    row_sums = train_matrix.sum(axis=1)
    new_matrix = train_matrix / row_sums
    return np.asarray(new_matrix)
