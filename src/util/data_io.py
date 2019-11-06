import os

import pandas as pd
import scipy

from src.util import log_utils as log
from src.util.io import load_pickle


def get_dataset(dataset_name, data_dir, logging=True):
    """
    Returns train, val and test data for a dataset. They are in COO form, as required by the Global, Personal, Mixture,
    NMF and LDA models.

    :param dataset_name: Name of directory where the train, val and test files are
    :param data_dir: Name of the directory the dataset is located.
    :param logging: Boolean that defines whether to print out the logs.
    :return: three COO arrays, corresponding to train, val and test.
    """
    if logging:
        log.info('Loading data for %s' % dataset_name)
    cols = ['UID', 'PID', 'cnt']
    df_train = pd.read_csv(os.path.join(data_dir, dataset_name, 'train.csv'))[cols]
    df_val = pd.read_csv(os.path.join(data_dir, dataset_name, 'validation.csv'))[cols]
    df_test = pd.read_csv(os.path.join(data_dir, dataset_name, 'test.csv'))[cols]
    return df_train.values, df_val.values, df_test.values


def get_decay_dataset(dataset_name, data_dir, decay, is_eval=True, logging=True):
    """
    Returns train, val and test data for a dataset. They are in COO form, as required by the Global, Personal, Mixture,
    NMF and LDA models.

    :param dataset_name: Name of directory where the train, val and test files are.
    :param data_dir: Name of the directory the dataset is located.
    :param decay: Decay factor.
    :param is_eval: Boolean, that defines if this data will be used for evaluation or training. The difference is that
    when evaluating, the validation data is added to the training, otherwise it is not.
    :param logging: Boolean that defines whether to print out the logs.
    :return: three COO arrays, corresponding to train, val and test.
    """
    if logging:
        log.info('Loading data for %s' % dataset_name)
    cols = ['UID', 'PID', 'cnt']
    df_train = pd.read_csv(os.path.join(data_dir, dataset_name, 'train.csv'))
    df_val = pd.read_csv(os.path.join(data_dir, dataset_name, 'validation.csv'))[cols]
    df_test = pd.read_csv(os.path.join(data_dir, dataset_name, 'test.csv'))[cols]
    for col in cols:
        df_val[col] = df_val[col].astype(int)
        df_test[col] = df_test[col].astype(int)

    if is_eval:
        # Apply decay on the last baskets in training data
        df_train['decay_cnt'] = decay ** (df_train['decay'] + 1) * df_train['cnt']
    else:
        # Do not apply decay on the last baskets in training data
        df_train['decay_cnt'] = decay ** (df_train['decay']) * df_train['cnt']

    for col in ['UID', 'PID']:
        df_train[col] = df_train[col].astype(int)
    df_train = df_train[['UID', 'PID', 'decay_cnt']]
    return df_train.values, df_val.values, df_test.values


def user_item_mat(d, path):
    file_name = d + '.csv'
    df = pd.read_csv(os.path.join(path, file_name))
    sparse_mat = scipy.sparse.coo_matrix((df['cnt'], (df['UID'], df['PID'])))
    return sparse_mat


def get_data_mat(dataset, data_dir):
    path = os.path.join(data_dir, dataset)
    return [user_item_mat(d, path) for d in ['train', 'validation', 'test']]


def get_hpf_dataset(dataset_name, data_dir, logging=True):
    """
    Returns train, val and test data for a dataset. They are in DataFrame form, as required by the HPF model.

    :param dataset_name: Name of directory where the train, val and test files are
    :param data_dir: Name of the directory the dataset is located.
    :param logging: Boolean that defines whether to print out the logs.
    :return: three DataFrame objects, corresponding to train, val and test
    """

    if logging:
        log.info('Loading data for %s' % dataset_name)
    cols = ['UID', 'PID', 'cnt']
    df_train = pd.read_csv(os.path.join(data_dir, dataset_name, 'train.csv'))[cols]
    df_train = df_train[df_train['cnt'] > 0]
    df_val = pd.read_csv(os.path.join(data_dir, dataset_name, 'validation.csv'))[cols]
    df_test = pd.read_csv(os.path.join(data_dir, dataset_name, 'test.csv'))[cols]
    return [df.reset_index(drop=True).rename(columns={'UID': 'UserId', 'PID': "ItemId", 'cnt': 'Count'}) for df in
            [df_train, df_val, df_test]]


def get_fpmc_dataset(dataset, data_dir, is_eval=True):
    # :param is_eval: Boolean, that defines if this data will be used for evaluation or training. The difference is that
    # when evaluating, the validation data is added to the training, otherwise it is not.
    tr_data = pd.read_csv(os.path.join(data_dir, dataset, 'fpmc_train.csv'))
    if is_eval:
        val_data = pd.read_csv(os.path.join(data_dir, dataset, 'fpmc_validation.csv'))
        tr_data = pd.concat([tr_data, val_data], axis=0)
        test_data = pd.read_csv(os.path.join(data_dir, dataset, 'fpmc_validation.csv'))
        test_data['prev_bsk'] = test_data['prev_bsk'].apply(fpmc_conform)
        te_data = test_data[['UID', 'prev_bsk']].drop_duplicates().values
    else:
        val_data = pd.read_csv(os.path.join(data_dir, dataset, 'fpmc_validation.csv'))
        val_data['prev_bsk'] = val_data['prev_bsk'].apply(fpmc_conform)
        te_data = val_data[['UID', 'prev_bsk']].drop_duplicates().values

    tr_data['prev_bsk'] = tr_data['prev_bsk'].apply(fpmc_conform)
    tr_data = tr_data.values

    test = get_fpmc_test(dataset, data_dir, is_eval)
    obj = load_pickle(os.path.join(data_dir, dataset, 'mapping.pkl'))
    user_set = set(obj['user'].values())
    item_set = set(obj['item'].values())
    input_data = [tr_data, te_data, test, user_set, item_set]
    return input_data


def get_fpmc_test(dataset, data_dir, is_eval=True):
    # :param is_eval: Boolean, that defines if this data will be used for evaluation or training. The difference is that
    # when evaluating, the validation data is added to the training, otherwise it is not.
    if is_eval:
        file_name = 'test.csv'
    else:
        file_name = 'validation.csv'
    cols = ['UID', 'PID', 'cnt']
    df_test = pd.read_csv(os.path.join(data_dir, dataset, file_name))[cols]
    for col in cols:
        df_test[col] = df_test[col].astype(int)
    return df_test.values


def fpmc_conform(s):
    return tuple([int(i) for i in s[1:-1].split(',') if i != ''])
