#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:09:44 2020
@author: krishna
"""

import os
import numpy as np
import glob
import argparse


def create_meta(files_list, store_loc, mode='train'):
    if not os.path.exists(store_loc):
        os.makedirs(store_loc)

    if mode == 'train_label':
        meta_store = store_loc + '/training_label.txt'
        fid = open(meta_store, 'w')
        for filepath in files_list:
            fid.write(filepath + '\n')
        fid.close()
    elif mode == 'train':
        meta_store = store_loc + '/training.txt'
        fid = open(meta_store, 'w')
        for filepath in files_list:
            fid.write(filepath + '\n')
        fid.close()
    elif mode == 'test':
        meta_store = store_loc + '/testing.txt'
        fid = open(meta_store, 'w')
        for filepath in files_list:
            fid.write(filepath + '\n')
        fid.close()
    elif mode == 'test_label':
        meta_store = store_loc + '/testing_label.txt'
        fid = open(meta_store, 'w')
        for filepath in files_list:
            fid.write(filepath + '\n')
        fid.close()
    else:
        print('Error in creating meta files')


def extract_files(folder_path):
    # audio_path = glob.glob(folder_path)
    # all_audio_path = glob.glob(os.path.dirname(folder_path+'/*.wav'))
    all_lang_folders = glob.glob(folder_path + '/*/')
    train_nums = len(all_lang_folders) - int(len(all_lang_folders) * 0.1)
    print(train_nums)
    test_nums = len(all_lang_folders) - train_nums
    train_lists = []
    test_lists = []
    # lang_folderpath = '/share/home/yandiqun/yz/TIMIT_/data/TEST/DR5/FHEW0/'
    for i, lang_folderpath in enumerate(all_lang_folders[:train_nums]):
        # person = 'FAKS0'
        # person = lang_folderpath.split('/')[-2]
        sub_folders = glob.glob(lang_folderpath+'/*.wav')
        # audio_filepath = '/share/home/yandiqun/yz/TIMIT_/data/TEST/DR1/FAKS0/SA1_.wav'
        for audio_filepath in sub_folders:
            to_write = audio_filepath + ' ' + str(i)
            print(to_write)
            train_lists.append(to_write)

    for i, lang_folderpath in enumerate(all_lang_folders[train_nums:]):
        # person = 'FAKS0'
        # person = lang_folderpath.split('/')[-2]
        sub_folders = glob.glob(lang_folderpath+'/*.wav')
        # audio_filepath = '/share/home/yandiqun/yz/TIMIT_/data/TEST/DR1/FAKS0/SA1_.wav'
        for audio_filepath in sub_folders:
            to_write = audio_filepath + ' ' + str(i)
            test_lists.append(to_write)

#好像这样取人的标签比较难，所以明天试试用源代码，赋标签。

    # train_lists = []
    # train_label_lists = []
    # test_lists = []
    # test_label_lists = []
    # for i, folder in enumerate(audio_path[:train_nums]):
    #     to_write = os.path.join(folder)
    #     train_lists.append(to_write)
    #
    # for i, folder in enumerate(audio_path[train_nums:]):
    #     to_write = os.path.join(folder)
    #     test_lists.append(to_write)

    # for i, folder in enumerate(all_audio_path[:train_nums]):
    #     to_write = os.path.join(folder, str(i))
    #     train_label_lists.append(to_write)
    #
    # for i, folder in enumerate(all_audio_path[train_nums:]):
    #     to_write = os.path.join(folder, str(i))
    #     test_label_lists.append(to_write)

    print("all_train_nums:%d all_test_nums:%d" % (train_nums, test_nums))
    return train_lists, test_lists


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--processed_data", default=r"F:\TIMIT_/*/*/*", type=str,
                        help='Dataset path')
    parser.add_argument("--meta_store_path", default="meta/", type=str, help='Save directory after processing')
    config = parser.parse_args()
    train_list, test_list = extract_files(config.processed_data)

    create_meta(train_list, config.meta_store_path, mode='train')
    # create_meta(train_label_list, config.meta_store_path, mode='train_label')
    create_meta(test_list, config.meta_store_path, mode='test')
    # create_meta(test_label_list, config.meta_store_path, mode='test_label')
