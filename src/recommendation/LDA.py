import os

from sklearn.decomposition import LatentDirichletAllocation

from src.recommendation.mixture_functions import get_train, get_test, get_val
from src.util.io import load_pickle, save_pickle

model_type = 'lda_model'


def train_lda_model(train, val, test, results_dir, dataset_name, overwrite=False, save_multinomials=True,
                    n_components=50, is_tune=True):
    """
    Runs the LDA experiment with scikit-learn implementation. It evaluates on the test set.

    Data come in COO form. That is a numpy array of (N x 3) where each row is the (row, column, value) triplet of the
    sparse array UID x PID. N is the number of entries in the array.

    :param train: Train data COO matrix.
    :param val: Validation data COO matrix.
    :param test: Test data COO matrix.
    :param results_dir: Name of the directory the results will be saved.
    :param dataset_name: Name of the directory the results will be saved.
    :param overwrite: Boolean, on whether to overwrite the multinomials or read them if they exist.
    :param save_multinomials: Boolean, on whether to save the multinomials.
    :param n_components: Number of components in LDA.
    :param is_tune: Boolean, that defines if this data will be used for evaluation or training. The difference is that
    when evaluating, the validation data is added to the training, otherwise it is not.
    :return: return multinomials, which is a dense matrix of predicted probability
    """

    filename = os.path.join(results_dir, model_type, dataset_name, str(n_components), 'user_multinomials.pkl')

    if os.path.exists(filename) and not overwrite:
        all_multinomials = load_pickle(filename)
    else:
        model = LatentDirichletAllocation(n_components=n_components, random_state=0)
        if is_tune:
            test_matrix = get_test(train, val, test)
        else:
            test_matrix = get_val(train, val, test)
        train_matrix = get_train(train, val, test, is_tune)
        model.fit(train_matrix)
        all_multinomials = model.transform(test_matrix).dot(model.components_)
        if save_multinomials:
            save_pickle(filename, all_multinomials, False)
    return all_multinomials
