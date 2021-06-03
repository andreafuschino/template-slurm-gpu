#! /usr/bin/env python3

import collections
import operator
import errno
import glob
import os
import itertools

import numpy as np
import torch
import yaml
import logging
import inspect
import datetime
import sqlite3
import tqdm
import tarfile, zipfile
from . import constants as const




def move_optimizer_to_gpu(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def makedir_if_not_there(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_yaml(fname):
    with open(fname, 'r') as f:
        loaded_yaml = yaml.safe_load(f)
    return loaded_yaml


def write_yaml(fname, input_dict, open_as):
    with open(fname, open_as) as outfile:
        yaml.dump(input_dict, outfile, default_flow_style=False, sort_keys=False)



def get_sorted_config_diff_folders(config_folder):
    full_base_path = os.path.join(config_folder, const.CONFIG_DIFF_BASE_FOLDER_NAME)
    config_diff_folder_names = glob.glob("%s*"%full_base_path) 
    latest_epochs = []
    if len(config_diff_folder_names) > 0:
        for c in config_diff_folder_names:
            latest_epochs.append([c]+[int(x) for x in c.replace(full_base_path,"").split('_')])
        num_training_sets = len(latest_epochs[0])-1
        latest_epochs = sorted(latest_epochs, key=operator.itemgetter(*list(range(1, num_training_sets+1))))
        return [x[0] for x in latest_epochs], [x[1:] for x in latest_epochs]
    return [], []

def get_all_resume_training_config_diffs(config_folder, split_manager):
    config_diffs, latest_epochs = get_sorted_config_diff_folders(config_folder)
    if len(config_diffs) == 0:
        return {}
    split_scheme_names = [split_manager.get_split_scheme_name(i) for i in range(len(latest_epochs[0]))]
    resume_training_dict = {}
    for i, k in enumerate(config_diffs):
        resume_training_dict[k] = {split_scheme:epoch for (split_scheme,epoch) in zip(split_scheme_names, latest_epochs[i])}
    return resume_training_dict



def get_last_linear(input_model, return_name=False):
    for name in ["fc", "last_linear"]:
        last_layer = getattr(input_model, name, None)
        if last_layer:
            if return_name:
                return last_layer, name
            return last_layer

def set_last_linear(input_model, set_to):
    setattr(input_model, get_last_linear(input_model, return_name=True)[1], set_to)


def check_init_arguments(input_obj, str_to_check):
    obj_stack = [input_obj]
    while len(obj_stack) > 0:
        curr_obj = obj_stack.pop()
        obj_stack += list(curr_obj.__bases__)
        if str_to_check in str(inspect.signature(curr_obj.__init__)):
            return True
    return False


def try_getting_db_count(record_keeper, table_name):
    try:
        len_of_existing_record = record_keeper.query("SELECT count(*) FROM %s"%table_name, use_global_db=False)[0]["count(*)"] 
    except sqlite3.OperationalError:
        len_of_existing_record = 0
    return len_of_existing_record


def get_datetime():
    return datetime.datetime.now()


def extract_progress(compressed_obj):
    logging.info("Extracting dataset")
    if isinstance(compressed_obj, tarfile.TarFile):
        iterable = compressed_obj
        length = len(compressed_obj.getmembers())
    elif isinstance(compressed_obj, zipfile.ZipFile):
        iterable = compressed_obj.namelist()
        length = len(iterable)
    for member in tqdm.tqdm(iterable, total=length):
        yield member


def if_str_convert_to_singleton_list(input):
    if isinstance(input, str):
        return [input]
    return input

def first_key_of_dict(input):
    return list(input.keys())[0]

def first_val_of_dict(input):
    return input[first_key_of_dict(input)]


def get_attr_and_try_as_function(input_object, input_attr):
    attr = getattr(input_object, input_attr)
    try:
        return attr()
    except TypeError:
        return attr


def get_eval_record_name_dict(hooks, tester, split_names=None):
    prefix = hooks.record_group_name_prefix 
    hooks.record_group_name_prefix = "" #temporary
    if split_names is None:
        non_meta = {"base_record_group_name": hooks.base_record_group_name(tester)}
    else:
        non_meta = {k:hooks.record_group_name(tester, k) for k in split_names}
    hooks.record_group_name_prefix = prefix
    return non_meta