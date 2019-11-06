import os

import numpy as np
from hpfrec import HPF
from scipy.sparse import coo_matrix

from src.util.io import load_pickle, save_pickle

model_type = 'hpf_model'


def train_hpf_model(train, val, test, results_dir, dataset_name, overwrite=False, save_multinomials=True,
                    n_components=500, top_n=10, stop_crit='val-llk', maxiter=500, ):
    """
    Runs the HPF experiment with hpfrec implementation. It evaluates on the test set.

    Data come in COO form. That is a numpy array of (N x 3) where each row is the (row, column, value) triplet of the
    sparse array UID x PID. N is the number of entries in the array.

    :param train: Train data COO matrix.
    :param val: Validation data COO matrix.
    :param test: Test data COO matrix.
    :param results_dir: Name of the directory the results will be saved.
    :param dataset_name: Name of the directory the results will be saved.
    :param overwrite: Boolean, on whether to overwrite the multinomials or read them if they exist.
    :param save_multinomials: Boolean, on whether to save the multinomials.
    :param n_components: Number of components in HPF.
    :param top_n: min number of predictions per user. Higher number requires more computational power. If set as False,
    all items are predicted for all users.
    :param stop_crit: Stopping criteria for the opimization procedure in HPF.
    :param maxiter: Maximum number of iterations for which to run the optimization procedure. This corresponds to epochs
    when fitting in batches of users.
    :return: return multinomials, which is a dense matrix of predicted probability
    """
    filename = os.path.join(results_dir, model_type, dataset_name, str(n_components), 'user_multinomials.pkl')
    if os.path.exists(filename) and not overwrite:
        all_multinomials = load_pickle(filename)
    else:
        model = HPF(k=n_components, stop_crit=stop_crit, maxiter=maxiter)
        model.fit(train, val_set=val)
        # item and user index starts from zero
        train_items = sorted(train['ItemId'].unique())
        num_items = train['ItemId'].max() + 1  # or len(train_items)
        # in case that users are different in train and test
        train_users = sorted(train['UserId'].unique())
        num_users = len(train_users)
        test_users = set(test['UserId'])

        pred_item = np.array([]).astype(int)
        pred_user = np.array([]).astype(int)
        pred_prob = np.array([]).astype(float)
        if top_n:
            for u in train_users:
                if u in test_users:
                    # Warning: nDCG and AUC cannot be evaluated
                    top_items = model.topN(user=u, n=top_n, exclude_seen=False)
                    user_id = np.full(shape=top_n, fill_value=u, dtype=np.int)
                    prob = model.predict(user=[u] * top_n, item=list(top_items))
                    pred_item = np.append(pred_item, top_items)
                    pred_user = np.append(pred_user, user_id)
                    pred_prob = np.append(pred_prob, prob)
                else:
                    continue
            all_multinomials = coo_matrix((pred_prob, (pred_user, pred_item)), shape=(num_users, num_items)).toarray()
        else:
            all_multinomials = np.zeros((num_users, num_items))  # train_items or test_items
            for i, u in enumerate(train_users):
                array_1d = model.predict(user=[u] * num_items, item=train_items)
                all_multinomials[i, :] = array_1d

        if save_multinomials:
            save_pickle(filename, all_multinomials, False)

    return np.nan_to_num(all_multinomials)
