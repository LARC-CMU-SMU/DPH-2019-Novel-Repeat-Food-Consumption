import os

import numpy as np
import pandas as pd

from src.recommendation.evaluation import print_evaluation
from src.util.array_util import data_to_sparse
from src.util.data_io import get_dataset, get_decay_dataset, get_hpf_dataset, get_fpmc_dataset
from src.util.io import load_pickle, save_pickle


def model_results(datasets, train_model, model_type, data_dir, results_dir, eval_lst, overwrite=False,
                  save_multinomials=True, save_model=False, is_eval=True, n_components=50, top_n=10, n_epoch=150,
                  regular=0.001, n_neg=10, lr=0.01, new_items=False, filename=False, ):
    """ Method that trains and evaluates for a model type with all dataset.

    :param datasets: List of dataset names.
    :param train_model: Model function to be used.
    :param model_type: Name of the model.
    :param data_dir: Name of the directory the data will be retrieved from.
    :param results_dir: Name of the directory the results will be saved.

    :param n_components: Number of components in FPMC.
    :param n_epoch: Number of epochs iterated.
    :param regular: Degree of regularization.
    :param n_neg: Number of negative samples per batch.
    :param lr: The learning rate.
    :param overwrite: Boolean, on whether to overwrite the multinomials or read them if they exist.
    :param save_multinomials: Boolean, on whether to save the multinomials.
    :param save_model: Boolean, on whether to save the model parameters.

    :return: return DataFrame output
    """
    if not filename:
        if new_items:
            filename = os.path.join(results_dir, '_'.join([model_type, 'eval_results_new_items.pkl']))
        else:
            filename = os.path.join(results_dir, '_'.join([model_type, 'eval_results.pkl']))

    if os.path.exists(filename) and not overwrite:
        output = load_pickle(filename)
        return output
    else:
        results = []

    for dataset in datasets:
        if isinstance(n_components, (dict,)):
            n_compo = n_components[dataset]
        else:
            n_compo = n_components

        if model_type in ['global_model', 'personal_model']:
            train, val, test = get_dataset(dataset, data_dir)
            all_multinomials = train_model(train, val, test, results_dir, dataset, overwrite,
                                           save_multinomials, is_eval)
        elif model_type in ['nmf_model', 'lda_model']:
            train, val, test = get_dataset(dataset, data_dir)
            all_multinomials = train_model(train, val, test, results_dir, dataset, overwrite, save_multinomials,
                                           n_compo, is_eval)
        elif model_type == 'mixture_model':
            train, val, test = get_dataset(dataset, data_dir)
            all_multinomials = train_model(train, val, test, results_dir, dataset, overwrite, save_multinomials)
        elif model_type == 'mixture_decay_model':
            train, val, test = get_decay_dataset(dataset, data_dir, n_compo)
            all_multinomials = train_model(train, val, test, results_dir, dataset, overwrite, save_multinomials,
                                           model_type=model_type)

        elif model_type == 'hpf_model':
            train, val, test = get_hpf_dataset(dataset, data_dir)
            all_multinomials = train_model(train, val, test, results_dir, dataset, overwrite, save_multinomials,
                                           int(n_compo), top_n)
            test = test.values
        elif model_type == 'fpmc_model':
            input_data = get_fpmc_dataset(dataset, data_dir)
            train, val, test = get_dataset(dataset, data_dir)
            all_multinomials = train_model(input_data, results_dir, dataset, n_components, n_epoch, regular, n_neg,
                                           lr, overwrite, save_multinomials, save_model)

            print("Unknown model ", model_type)
            return None

        if new_items:
            prev_items = data_to_sparse(np.vstack((train, val))).todense()
            all_multinomials[np.where(prev_items != 0)] = 0

        result = _results(dataset, eval_lst, test, all_multinomials)
        results.append(result)

    output = format_results(results)
    save_pickle(filename, output, False)

    return output


def _results(dataset, eval_lst, test, all_multinomials):
    sparse_test = data_to_sparse(test)
    eval_result = []
    for _method, k_vals in eval_lst:
        for k in k_vals:
            method, value = print_evaluation(test, sparse_test, all_multinomials, _method, k)
            eval_result.append([method, value])
    return [dataset, eval_result]


def format_results(results):
    header = ['dataset'] + ordered_set([item[0] for sublist in results for item in sublist[1]])
    output = pd.DataFrame([[line[0]] + [pair[1] for pair in line[1]]
                           for line in results], columns=header).set_index('dataset')
    return output


def ordered_set(x):
    return sorted(set(x), key=x.index)
