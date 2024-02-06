# filter zero123 generated views from objaverse, filter out invalid images that are pure white

import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import argparse

def filter_zero123_views(path):
    invalid_ids = []
    objects = os.listdir(path)
    for obj in tqdm(objects):
        views = glob.glob(os.path.join(path, obj, '*.png'))
        # check if the number of views is 12
        if len(views) != 12:
            invalid_ids.append(obj)
            print(obj, 'empty')
            continue
        # read image and check if it is pure white
        invalid = 0
        for view in views:
            img = plt.imread(view)
            if np.all(img[:, :, -1] == 0.):
                invalid += 1

        if invalid >= 3:
            invalid_ids.append(obj)
            print(obj, 'invalid')

    return invalid_ids

def move_invalid_views(path, invalid_ids, invalid_path):
    for obj_id in tqdm(invalid_ids):
        # if exist, remove
        if os.path.exists(os.path.join(path, obj_id)):
            # move folder to invalid folder
            shutil.move(os.path.join(path, obj_id), os.path.join(invalid_path, obj_id))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter & Move Zero-1-to-3 Objaverse Rendering Data.")
    parser.add_argument(
        "--path",
        type=str,
        default="/data/objaverse/views_release",
        required=True,
        help="Path to Zero-1-to-3 Objaverse views_release Rendering Data.",
    )
    args = parser.parse_args()
    path = args.path

    # # filter invalid views
    # invalid_ids = filter_zero123_views(path)
    # # save invalid ids
    # np.save('invalid_ids.npy', invalid_ids)
    # # print(invalid_ids)
    # print("Total invalid len ", len(invalid_ids))

    # move invalid views
    invalid_ids = np.load('all_invalid.npy')
    invalid_path = os.path.join(path, '../invalid')
    os.makedirs(invalid_path, exist_ok=True)
    move_invalid_views(path, invalid_ids, invalid_path)