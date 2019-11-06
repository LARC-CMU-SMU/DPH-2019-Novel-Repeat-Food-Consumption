import numpy as np
from numba import jit

from src.util.ndcg import ndcg_score


def print_evaluation(test, sparse_test, all_multinomials, method='logP', k='', _print=True):
    """
    Evaluates the model (resulted multinomials) on the test data, based on the method selected.
    Prints evaluation metric (averaged across users).

    :param test: COO matrix with test data.
    :param all_multinomials: matrix of probability estimates of each user, item
    :param method: string, can be 'logp' for log probability, 'recall',
    'precision', 'npk' 'ndcg', 'auc', 'prc_auc', 'micro_precision'
    :param k: the k from recall@k, precision@k or nDCG@k. If method is logP, then this does nothing.
    :param _print: whether the information is printed
    :return: method, score, and standard deviation if applicable
    """

    per_event = evaluate(test, sparse_test, all_multinomials, method, k)

    if k == '':
        _method = method
    elif k:
        _method = method + '@' + str(k)
    else:
        _method = method + '@' + 'M'

    if _print:
        print('%s: %.5f' % (_method, per_event))
    return _method, per_event


def evaluate(test_data, sparse_test, all_multinomials, method, k):
    """
    Evaluates the model (resulted multinomials) on the test data, based on the method selected.

    :param test_data: COO matrix with test data.
    :param all_multinomials: matrix of probability estimates of each user, item
    :param method: string, can be 'logp' for log probability, 'recall',
    'precision', 'npk' 'ndcg', 'auc', 'prc_auc', 'micro_precision'
    :param k: the k from recall@k, various version of precision@k or nDCG@k.
    If method is logP, 'auc', 'prc_auc' then this does nothing.
    :return: score, and standard deviation if applicable
    """

    if method.lower() == 'logp':
        test_points = np.repeat(test_data[:, :-1], test_data[:, -1].astype(int), axis=0).astype(int)
        test_probs = all_multinomials[list(test_points.T)]
        return np.mean(np.log(test_probs))
    elif method.lower() == 'recall':
        return recall_at_top_k(sparse_test, all_multinomials, k)
    elif method.lower() == 'precision':
        return precision_at_top_k(sparse_test, all_multinomials, k)
    elif method.lower() == 'ndcg':
        return avg_ndcg(sparse_test, all_multinomials, k)
    elif method.lower() == 'user_ndcg':
        return avg_ndcg_users(sparse_test, all_multinomials, k)
    else:
        print('I do not know this evaluation method')


@jit(nogil=True)
def rates_to_exp_order(rates, argsort, exp_order, M):
    prev_score = 0
    prev_idx = 0
    prev_val = rates[argsort[0]]
    for i in range(1, M):
        if prev_val == rates[argsort[i]]:
            continue

        tmp = 0
        for j in range(prev_idx, i):
            exp_order[argsort[j]] = prev_score + 1
            tmp += 1

        prev_score += tmp
        prev_val = rates[argsort[i]]
        prev_idx = i

    # For the last equalities
    for j in range(prev_idx, i + 1):
        exp_order[argsort[j]] = prev_score + 1


@jit(nogil=True)
def rates_mat_to_exp_order(rates, argsort, exp_order, N, M):
    for i in range(N):
        rates_to_exp_order(rates[i], argsort[i], exp_order[i], M)


# @jit(cache=True)
@jit
def fix_exp_order(rates, exp_order, k, N):
    for i in range(N):
        mask = np.where(exp_order[i] <= k)[0]
        if len(mask) <= k:
            exp_order[i] = 0
            exp_order[i, mask] = 1
        else:
            max_val = np.max(exp_order[i, mask])
            max_val_mask = np.where(exp_order[i] == max_val)[0]
            exp_order[i] = 0
            exp_order[i, mask] = 1
            exp_order[i, max_val_mask] = (k - max_val + 1) / max_val_mask.shape[0]


def fix_exp_order_diff_k(rates, exp_order, k, N):
    # k is array of the same length of N
    for i in range(N):
        mask = np.where(exp_order[i] <= k[i])[0]
        if len(mask) <= k[i]:
            exp_order[i] = 0
            exp_order[i, mask] = 1
        else:
            max_val = np.max(exp_order[i, mask])
            max_val_mask = np.where(exp_order[i] == max_val)[0]
            exp_order[i] = 0
            exp_order[i, mask] = 1
            exp_order[i, max_val_mask] = (k[i] - max_val + 1) / max_val_mask.shape[0]


@jit
def recall_at_top_k(test_counts, scores, k):
    """
    Compute recall at k, Kotzias's version with frequency as weights

    :param test_counts: COO matrix with test data.
    :param scores: matrix of probability estimates of each user, item
    :param k: The rank k from nDCG@k.
    :return: average score across users and the standard deviation
    """
    argsort = np.argsort(-scores, axis=1)
    exp_order = np.zeros(scores.shape)

    rates_mat_to_exp_order(scores, argsort, exp_order, scores.shape[0], scores.shape[1])
    fix_exp_order(scores, exp_order, k, scores.shape[0])
    recall_in = test_counts.multiply(exp_order)
    u_recall = recall_in.sum(axis=1) / test_counts.sum(axis=1)  # gives runtime warning, but its ok. NaN are handled.
    u_recall = u_recall[~np.isnan(u_recall)]  # nan's do not count.
    return np.mean(u_recall)


@jit
def precision_at_top_k(test_counts, scores, k):
    """
    Compute precision at k.

    :param test_counts: COO matrix with test data.
    :param scores: matrix of probability estimates of each user, item
    :param k: The rank k from nDCG@k.
    :return: average score across users and the standard deviation
    """

    # obtain index order by high to low scores
    argsort = np.argsort(-scores, axis=1)
    # initialize exp_order with all 0s
    exp_order = np.zeros(scores.shape)
    # update exp_order: rank of item by predicted score for the user
    rates_mat_to_exp_order(scores, argsort, exp_order, scores.shape[0], scores.shape[1])
    # update exp_order: item is in top-k recommendations for the user
    fix_exp_order(scores, exp_order, k, scores.shape[0])
    # numerator in precision computation
    test_counts = test_counts.astype(bool).astype(int).astype(float)
    precision_in = test_counts.multiply(exp_order)
    u_precision = precision_in.sum(axis=1) / k
    # u_precision = u_precision[~np.isnan(u_precision)]
    return np.mean(u_precision)


@jit
def avg_ndcg(test_data, all_multinomials, k):
    """
    Compute nDCG.

    :param test_data: COO matrix with test data.
    :param all_multinomials: matrix of probability estimates of each user, item
    :param k: The rank k from nDCG@k.
    :return: average score across users and the standard deviation
    """
    dense_test = test_data.todense()
    scores = []
    for i, c in enumerate(dense_test):
        score = ndcg_score(c, [all_multinomials[i]], k=k, ignore_ties=False)
        scores.append(score)
    return np.mean(scores)


def avg_ndcg_users(test_data, all_multinomials, k):
    """
    Compute nDCG.

    :param test_data: COO matrix with test data.
    :param all_multinomials: matrix of probability estimates of each user, item
    :param k: The rank k from nDCG@k.
    :return: average score across users and the standard deviation
    """
    dense_test = test_data.todense()
    scores = []
    for i, c in enumerate(dense_test):
        score = ndcg_score(c, [all_multinomials[i]], k=k, ignore_ties=False)
        scores.append(score)
    return scores
