import os

import numpy as np

from src.recommendation.mixture_functions import get_train_global, learn_individual_mixing_weights, \
    learn_global_mixture_weights
from src.util.array_util import data_to_sparse
from src.util.io import load_pickle, save_pickle


def train_mixture_model(train, val, test, results_dir, dataset_name, overwrite=False, save_multinomials=True,
                        num_proc=None, model_type='mixture_model'):
    """
    Runs the Mixture experiment with the original paper implementation. It evaluates on the test set.
    See https://github.com/UCIDataLab/repeat-consumption.

    Data come in COO form. That is a numpy array of (N x 3) where each row is the (row, column, value) triplet of the
    sparse array UID x PID. N is the number of entries in the array.

    :param train: Train data COO matrix.
    :param val: Validation data COO matrix.
    :param test: Test data COO matrix.
    :param results_dir: Name of the directory the results will be saved.
    :param dataset_name: Name of the directory the results will be saved.
    :param overwrite: Boolean, on whether to overwrite the multinomials or read them if they exist.
    :param save_multinomials: Boolean, on whether to save the multinomials.
    :param num_proc: Number of processes to be used. If none, all the processors in the machine will be used.
    :return: return multinomials, which is a dense matrix of predicted probability
    """
    filename = os.path.join(results_dir, model_type, dataset_name, 'user_multinomials.pkl')
    if os.path.exists(filename) and not overwrite:
        user_multinomials = load_pickle(filename)
    else:
        weight_filename = os.path.join(results_dir, 'mixture_model', dataset_name, 'mixing_weights.pkl')
        if os.path.exists(weight_filename) and not overwrite:
            # an array of mixing weights, which is n_users x 2 (2 components, self and global)
            mix_weights = load_pickle(weight_filename)
        else:
            train_matrix, global_matrix = get_train_global(train, val, test)
            components = [train_matrix, global_matrix]  # can add more components here
            mix_weights = learn_mixing_weights(components, val, num_proc)
            if save_multinomials:
                save_pickle(weight_filename, mix_weights, False)

        user_multinomials = _user_multinomials(train, val, test, mix_weights)

        if save_multinomials:
            save_pickle(filename, user_multinomials, False)

    return user_multinomials


def learn_mixing_weights(components, validation_data, num_proc=None):
    """Runs the Smoothed Mixture model on the number of components. Each component is an array of UID x PID
    (user x items). This runs in parallel for efficiency.

    :param components: List of matrices (CSR or full) -- all must have the same size.
    :param validation_data: COO matrix of validation data.
    :param num_proc: Number of processes to be used. If none, all the processors in the machine will be used.
    :return return the mixing weights for each user (or the entire population).
    """
    alpha = 1.001  # very small prior for global.
    global_mix_weights = learn_global_mixture_weights(alpha, components, validation_data)  # learn global mix weights.
    val_data = data_to_sparse(validation_data)
    # use global mixing weights as prior for individual ones.
    user_mix_weights = learn_individual_mixing_weights(global_mix_weights, components, val_data, num_proc)
    return user_mix_weights


def _user_multinomials(train, val, test, mix_weights):
    """
    Predicts the scores of users for items.
    :param train: Train data COO matrix.
    :param val: Validation data COO matrix.
    :param test: Test data COO matrix.
    :param mix_weights: the mixing weights for each user (or the entire population).
    :return: return multinomials, which is a dense matrix of predicted probability
    """
    eval_train_matrix, eval_glb_matrix = get_train_global(train, val, test, is_eval=True)
    eval_components = [eval_train_matrix, eval_glb_matrix]
    user_multinomials = mix_multinomials(eval_components, mix_weights)
    return user_multinomials


def mix_multinomials(components, mixing_weights):
    """Returns the multinomial distribution for each user after mixing them.

    :param components: List of matrices (CSR or full) -- all must have the same size.
    :param mixing_weights: List of mixing weights (for each component). If mixing weights is an array, then it means
    there is one mixing weight per user. Otherwise, they are global.
    :return: return multinomials, which is a dense matrix of predicted probability
    """
    if mixing_weights.ndim == 2:
        return _user_individual_multinomial(components, mixing_weights)
    else:
        return _user_global_multinomials(components, mixing_weights)


def _user_global_multinomials(components, mixing_weights):
    """The mixing weights are the same for each user.
    :param components: List of components
    :param mixing_weights: List of mixing weights (one for each component)
    :return: return multinomials, which is a dense matrix of predicted probability
    """

    result = np.zeros(components[0].shape)
    for i, c in enumerate(components):
        result += mixing_weights[i] * c
    return np.array(result)


def _user_individual_multinomial(components, mixing_weights):
    """ The mixing weights are different for each user.
    :param components: List of components
    :param mixing_weights: List of mixing weights (for each component, and each user)
    :return: return multinomials, which is a dense matrix of predicted probability
    """
    result = np.zeros(components[0].shape)
    for i, c in enumerate(components):
        c = np.array(c.todense())
        result += mixing_weights[:, i][:, np.newaxis] * c
    return result
