import os
import warnings

from collections import namedtuple

import tqdm
import numpy as np
import json


def print_args(args):
    if isinstance(args, dict):
        items = args.items()
    else:
        items = args.__dict__.items()
    print("######################################  Arguments  ######################################")
    for k, v in items:
        print("{0: <30}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("#########################################################################################")


def save_args(args, save_dir, is_print_args=True):
    print("Save dir:", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if is_print_args:
        print_args(args)
    with open(os.path.join(save_dir, r'opt.json'), 'w') as f:
        json.dump(args.__dict__, f)


def load_args(filename, is_print_args=True):
    with open(filename, 'r') as f:
        args = json.load(f)
    if is_print_args:
        print_args(args)
    Args = namedtuple('Args', args.keys())
    args = Args(**args)
    return args

