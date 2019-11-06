import os

from src.recommendation.fpmc_functions import FPMC, data_to_3_list
from src.util.io import load_pickle, save_pickle

model_type = 'fpmc_model'


def train_fpmc_model(input_data, results_dir, dataset_name, n_components, n_epoch=100, regular=0.001, n_neg=10,
                     lr=0.01, overwrite=False, save_multinomials=True, save_model=False):
    """
    Runs the FPMC experiment. Adapted from https://github.com/khesui/FPMC} to allow for variable-sized baskets.

    :param input_data: tr_data, te_data, test, user_set, item_set.
    :param results_dir: Name of the directory the results will be saved.
    :param dataset_name: Name of the directory the results will be saved.
    :param n_components: Number of components in FPMC.
    :param n_epoch: Number of epochs iterated.
    :param regular: Degree of regularization.
    :param n_neg: Number of negative samples per batch.
    :param lr: The learning rate.
    :param overwrite: Boolean, on whether to overwrite the multinomials or read them if they exist.
    :param save_multinomials: Boolean, on whether to save the multinomials.
    :param save_model: Boolean, on whether to save the model parameters.
    :return: return multinomials, which is a dense matrix of predicted probability
    """

    filename = os.path.join(results_dir, model_type, dataset_name, str(n_components), 'user_multinomials_.pkl')
    if os.path.exists(filename) and not overwrite:
        all_multinomials = load_pickle(filename)
    else:
        all_multinomials, model = _train_fpmc_model(input_data, n_components, n_epoch, regular, n_neg, lr)
        if save_multinomials:
            save_pickle(filename, all_multinomials, False)
        if save_model:
            save_model_param(model, results_dir, model_type, dataset_name)
    return all_multinomials


def _train_fpmc_model(input_data, n_components, n_epoch, regular, n_neg, lr):
    tr_data, te_data, test, user_set, item_set = input_data
    fpmc = FPMC(n_user=max(user_set) + 1, n_item=max(item_set) + 1, n_factor=n_components, learn_rate=lr,
                regular=regular)
    fpmc.user_set = user_set
    fpmc.item_set = item_set
    fpmc.init_model()
    tr_3_list = data_to_3_list(tr_data)
    for epoch in range(n_epoch):
        fpmc.learn_epoch(tr_3_list, n_neg)
    multinomials = fpmc.construct_multinomials(te_data)
    return multinomials, fpmc


def save_model_param(model, results_dir, model_type, dataset_name):
    for component, item in {'VUI': model.VUI, 'VIU': model.VIU, 'VLI': model.VLI, 'VIL': model.VIL}.items():
        filename = os.path.join(results_dir, model_type, dataset_name, component + '.pkl')
        save_pickle(filename, item, False)
