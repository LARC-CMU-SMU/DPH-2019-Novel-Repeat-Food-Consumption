import os

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.util.io import save_pickle


def data_split(data_dir=DATA_DIR, file_name='MFP.csv', t0=28, t1=7, t2=1, t3=1, t4=174,
               overwrite=False, _print=False, by_meal=False):
    df = pd.read_csv(os.path.join(data_dir, file_name))

    for d in np.arange(t0, t4, t3):
        if mapping_exist(d, data_dir) and not overwrite:
            pass
        else:
            _save_train_val_test(df, d, t1, t2, t3, data_dir, by_meal)


def _save_train_val_test(df, d1, t1, t2, t3, data_dir, by_meal):
    train, val, test = select_data(df, d1, t1, t2, t3)
    train, val, test, users, items = filter_all(train, val, test)
    user_mapping = {value: c for c, value in enumerate(sorted(users), 0)}
    item_mapping = {value: c for c, value in enumerate(sorted(items), 0)}
    dataset = map_dataset(train, val, test, user_mapping, item_mapping, by_meal)
    if test.shape[0] > 1:
        save_files(data_dir, d1, dataset, user_mapping, item_mapping, by_meal)


def select_data(df, d1, t1, t2, t3):
    train = df[df['days'].isin(range(d1, d1 + t1))]
    val = df[df['days'].isin(range(d1 + t1, d1 + t1 + t2))]
    test = df[df['days'].isin(range(d1 + t1 + t2, d1 + t1 + t2 + t3))]
    return train, val, test


def filter_all(train, val, test):
    users, items = _common_user_item(train, val, test)
    prev_n_users = len(users)
    prev_n_items = len(items)

    train, val, test, users, items, n_users, n_items = _filter_once(train, val, test, users, items)

    while prev_n_users != n_users or prev_n_items != n_items:
        prev_n_users = n_users
        prev_n_items = n_items
        train, val, test, users, items, n_users, n_items = _filter_once(train, val, test, users, items)

    return train, val, test, users, items


def _common_user_item(train, val, test):
    users = set(train['uid'].unique()) & set(val['uid'].unique()) & set(test['uid'].unique())
    items = set(train['food_id'].unique())
    return users, items


def _filter_user_item(train, val, test, users, items):
    train = train[(train['uid'].isin(users)) & (train['food_id'].isin(items))]
    val = val[(val['uid'].isin(users)) & (val['food_id'].isin(items))]
    test = test[(test['uid'].isin(users)) & (test['food_id'].isin(items))]
    return train, val, test


def _filter_once(train, val, test, users, items):
    train, val, test = _filter_user_item(train, val, test, users, items)
    users, items = _common_user_item(train, val, test)
    n_users = len(users)
    n_items = len(items)
    return train, val, test, users, items, n_users, n_items


def map_dataset(train, val, test, user_mapping, item_mapping, by_meal):
    lst = [_map_df(temp, user_mapping, item_mapping, by_meal) for
           temp in [train, val, test]]
    return lst


def _map_df(temp, user_mapping, item_mapping, by_meal):
    # last basket: 0, earlier basket with larger number
    seq_mapping = {d: i + 1 for i, d in enumerate(sorted(temp['days'].unique())[::-1])}
    temp.loc[:, 'decay'] = temp['days'].apply(lambda s: seq_mapping[s])
    temp.loc[:, 'UID'] = temp['uid'].apply(lambda s: user_mapping[s])
    temp.loc[:, 'PID'] = temp['food_id'].apply(lambda s: item_mapping[s])
    temp.loc[:, 'cnt'] = 1
    uid, pid = max(user_mapping.values()), max(item_mapping.values())
    if by_meal:
        temp = temp[['UID', 'PID', 'decay', 'cnt', 'days', 'meal']]
        # make shape consistent
        shape_df = pd.DataFrame.from_dict({'UID': [0] * 4 + [uid] * 4,
                                           'PID': [0] * 4 + [pid] * 4, 'decay': [-1] * 8, 'days': [0] * 8,
                                           'cnt': [0] * 8,
                                           'meal': ['breakfast', 'lunch', 'dinner', 'snack'] * 2})
    else:
        temp = temp[['UID', 'PID', 'decay', 'cnt', 'days']]
        # make shape consistent
        shape_df = pd.DataFrame.from_dict({'UID': [uid],
                                           'PID': [pid], 'decay': [-1], 'days': [0], 'cnt': [0], })
    temp = temp.append(shape_df, sort=False)
    return temp


def save_files(data_dir, d1, dataset, user_mapping, item_mapping, by_meal):
    file_names = ["train.csv", "validation.csv", "test.csv"]
    dest_dir = os.path.join(data_dir, 'day', str(d1))
    os.makedirs(dest_dir, exist_ok=True)
    if by_meal:
        for i, data in enumerate(dataset):
            for m in ['breakfast', 'lunch', 'dinner', 'snack']:
                dest_dir = os.path.join(data_dir, m, str(d1))
                os.makedirs(dest_dir, exist_ok=True)
                data_m = data[data['meal'] == m]
                data_m.to_csv(os.path.join(dest_dir, file_names[i]), index=False)
    else:
        for i, data in enumerate(dataset):
            data.to_csv(os.path.join(dest_dir, file_names[i]), index=False)
    save_mapping(user_mapping, item_mapping, dest_dir)


def save_mapping(user_mapping, item_mapping, dest_dir):
    obj = {'user': user_mapping, 'item': item_mapping}
    save_pickle(os.path.join(dest_dir, 'mapping.pkl'), obj)


def mapping_exist(d, data_dir):
    path = os.path.join(data_dir, str(d), "mapping.pkl")
    return os.path.exists(path)


def transform_fpmc_dataset(data_dir, dataset):
    df_val = pd.read_csv(os.path.join(data_dir, dataset, 'validation.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, dataset, 'test.csv'))
    df_train = pd.read_csv(os.path.join(data_dir, dataset, 'train.csv'))

    df_train.loc[:, 'flag'] = 'train'
    df_val.loc[:, 'flag'] = 'val'
    df_test.loc[:, 'flag'] = 'test'
    dfs = pd.concat([df_train, df_val, df_test])
    dfs1 = dfs[dfs['cnt'] != 0]
    tmp = dfs1.groupby(['UID', 'days'])['PID'].apply(tuple).to_dict()

    # {UID:{day:prev_day, }, }
    tmp1 = dfs1[['UID', 'days']].drop_duplicates()
    tmp1 = tmp1.groupby('UID')['days'].apply(list).to_frame()
    tmp1.loc[:, 'prev_day'] = tmp1['days'].apply(lambda li: dict(zip(li[1:], li)))
    prev_bsk = tmp1['prev_day'].to_dict()

    def find_prev(line, pid_lst=tmp, prev_basket=prev_bsk):
        prev_bsk_days = prev_basket[line['UID']].get(line['days'], 0)
        return pid_lst.get((line['UID'], prev_bsk_days), 0)

    dfs1.loc[:, 'prev_bsk'] = dfs1.apply(find_prev, axis=1)

    dfs0 = dfs1[dfs1['prev_bsk'] != 0]

    df_train_fpmc = dfs0[dfs0['flag'] == 'train'][['UID', "PID", 'prev_bsk']]
    df_val_fpmc = dfs0[dfs0['flag'] == 'val'][['UID', "PID", 'prev_bsk']]
    df_test_fpmc = dfs0[dfs0['flag'] == 'test'][['UID', "PID", 'prev_bsk']]

    df_train_fpmc.to_csv(os.path.join(data_dir, dataset, 'fpmc_train.csv'), index=False)
    df_val_fpmc.to_csv(os.path.join(data_dir, dataset, 'fpmc_validation.csv'), index=False)
    df_test_fpmc.to_csv(os.path.join(data_dir, dataset, 'fpmc_test.csv'), index=False)
