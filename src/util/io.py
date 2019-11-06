"""
This file contains methods that wrap around numpy and pickle for easy IO.
Save methods, create the directory path if needed and change the permissions to be shared by group.
"""
import csv
import os
import pickle
import platform

import numpy as np


def make_go_rw(filename, change_perm):
    if change_perm:
        if platform.python_version()[0] == '2':
            os.chmod(filename, 770)
        else:
            os.chmod(filename, 0o770)


def make_dir(filename):
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_pickle(filename, obj, other_permission=True):
    make_dir(filename)

    with open(filename, 'wb') as gfp:
        if platform.python_version()[0] == '2':
            pickle.dump(obj, gfp, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(obj, gfp, protocol=2)
        gfp.close()

    make_go_rw(filename, other_permission)


def load_pickle(filename):
    with open(filename, 'rb') as gfp:
        r = pickle.load(gfp, encoding='latin1')
    return r


def load_txt(filename, delimiter=','):
    d = np.loadtxt(filename, delimiter=delimiter)
    return d


def save_dict(mydict, file_name):
    with open(file_name, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in mydict.items():
            writer.writerow([key, value])
    print("saved: " + file_name)
