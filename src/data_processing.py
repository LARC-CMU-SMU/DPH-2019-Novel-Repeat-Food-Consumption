import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import DATA_DIR


def process_data(meta_data=False, date_1=u'2014-10-12', date_2=u'2015-03-14', max_kcal=3000):
    non_food = ['quick added calories',
                'quick add - quick add - 100 calories',
                'quick added kilojoules',
                'quick added calories - one calorie',
                '', ]

    non_food_cats = ['dietary_supplement',
                     'condiment',
                     'fast_food_brand',
                     'snack_brand',
                     'herb_spice',
                     'preparation']

    mfp = pd.read_csv(os.path.join(DATA_DIR, 'mfp_food_diaries.tsv'), sep='\t', low_memory=False)

    mfp = mfp[mfp['calories'] <= max_kcal]
    mfp = mfp[mfp['calories'] >= 0]
    n_users = mfp[['date', 'uid']].drop_duplicates().groupby('date').apply(len)
    days_selected = []
    for d in n_users.index:
        if (d >= date_1) and (d <= date_2):
            days_selected.append(d)
    mfp = mfp[mfp['date'].isin(days_selected)]

    ref = dict()
    for s in mfp["food"].unique():
        try:
            ref.update({s: process_food_name(s)})
        except:
            pass

    mfp["clean_food"] = mfp["food"].apply(lambda i: ref.get(i))
    mfp = mfp[~mfp["clean_food"].isin(non_food)]

    # remove records without categores
    mfp = mfp[mfp['categories'].notna()]

    def remove_category(s1):
        s1 = set(s1.split(', '))
        s1.difference_update(non_food_cats)
        if len(s1) > 0:
            return s1
        return np.nan

    mfp['categories'] = mfp['categories'].apply(remove_category)
    # remove items that belong to only some categories
    mfp = mfp[mfp['categories'].notna()]

    mfp = iterative_filtering(mfp)

    index_ref = {item: idx for idx, item in enumerate(sorted(mfp["clean_food"].unique()))}
    mfp["food_id"] = mfp["clean_food"].apply(lambda x: index_ref[x])
    days_ref = {d: count_days(d) for d in mfp['date'].unique()}
    mfp["days"] = mfp['date'].apply(lambda d: days_ref[d])

    date_file_path = os.path.join(os.path.join(DATA_DIR, 'days_date_mapping.npy'))
    np.save(date_file_path, days_ref)

    if meta_data:
        mfp.to_csv(os.path.join(DATA_DIR, 'MFP.meta.csv'), index=False)

    data_file_path = os.path.join(DATA_DIR, 'MFP.csv')
    mfp[['uid', 'days', 'food_id', 'meal']].to_csv(data_file_path, index=False)

    print("Total {:,} records.\n".format(mfp.shape[0]))


def count_days(d, date_format="%Y-%m-%d", start_date='2014-09-14'):
    return (datetime.strptime(d, date_format) - datetime.strptime(start_date, date_format)).days


def iterative_filtering(prev_df, item_threshold=20, user_threshold=5):
    prev_users_items = unique_users_items(prev_df)
    curr_df = filter_once(prev_df, item_threshold, user_threshold)
    curr_users_items = unique_users_items(curr_df)

    while curr_users_items != prev_users_items:
        prev_users_items = curr_users_items
        curr_df = filter_once(curr_df, item_threshold=20, user_threshold=5)
        curr_users_items = unique_users_items(curr_df)
    return curr_df


def filter_once(curr_df, item_threshold, user_threshold, food_col='clean_food', user_col='uid'):
    pairs = curr_df[[user_col, food_col]].drop_duplicates()
    # filter items
    item_frequency = pairs.drop_duplicates().groupby(food_col).apply(len)
    items_selected = item_frequency[item_frequency >= user_threshold].index
    # filter users
    user_frequency = pairs[pairs[food_col].isin(items_selected)].groupby(user_col).apply(len)
    users_selected = user_frequency[user_frequency >= item_threshold].index
    curr_df = curr_df[curr_df[food_col].isin(items_selected)]
    curr_df = curr_df[curr_df[user_col].isin(users_selected)]
    return curr_df


def unique_users_items(curr_df, food_col='clean_food', user_col='uid'):
    return curr_df[user_col].nunique(), curr_df[food_col].nunique()


def process_food_name(s1):
    # separators: ", " + any of (integer, decimal & fraction) +" "
    exp = r", \d+\.\d+ |, \d+\,\d+ |, \d+ |, \d+\/\d+ "
    # remove content in parenthesis for finding the separator
    if s1.count('(') == s1.count(')'):
        s2 = re.sub(r'[(].*?[\)]', ' ', s1)
    else:
        s2 = s1
    split_by = re.findall(exp, s2)[0]
    return clean_name(s1.split(split_by)[0])


def clean_name(name):
    name = name.replace("\t", " ").replace("\n", " ").replace("w/o", " no ").replace("w/", " ")
    return re.sub(' +', ' ', name.strip()).lower()


if __name__ == "__main__":
    process_data()
