import os
import json
import numpy as np
import pickle
import argparse
import random
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def compute_frequency(args, infodir):
    sample_names = [
        p for p in os.listdir(infodir)
        if os.path.isfile(os.path.join(infodir, p)) and 'json' in p
    ]

    if 'BP4D' in args.datasetdir:
        subject_names = np.unique(
            [sample_name[0:4] for sample_name in sample_names]).tolist()
    elif 'DISFA' in args.datasetdir:
        subject_names = np.unique(
            [sample_name[0:5] for sample_name in sample_names]).tolist()

    namelist = []
    for name in namelist:
        subject_names.remove(name)

    if 'BP4D' in args.datasetdir:
        # BP4D
        aulist_selected = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
        au_map = {
            1: 0,
            2: 1,
            4: 2,
            6: 4,
            7: 5,
            10: 7,
            12: 9,
            14: 11,
            15: 12,
            17: 14,
            23: 19,
            24: 20
        }
    elif 'DISFA' in args.datasetdir:
        # DISFA
        aulist_selected = [1, 2, 4, 6, 9, 12, 25, 26]
        au_map = {1: 0, 2: 1, 4: 2, 6: 4, 9: 5, 12: 6, 25: 10, 26: 11}

    cntsample = 0
    all_labelcnt = list()
    subject_labelcnt = np.zeros((len(subject_names), len(aulist_selected)))
    for sample_name in sample_names:
        sample_path = os.path.join(infodir, sample_name)
        with open(sample_path, 'r') as f:
            video_info = json.load(f)

        frame_infos = video_info['data']
        frame_infos = sorted(frame_infos,
                             key=lambda e: e['frame_index'],
                             reverse=False)

        cntframe = 0
        labelcnt = np.zeros(len(aulist_selected))
        for frame_info in frame_infos:

            label = list()
            for au in aulist_selected:
                label.append(int(frame_info['label'][au_map[au]]))
            label = np.array(label)

            labelcnt += label
            cntframe += 1
        all_labelcnt.append(labelcnt.tolist())
        for idx, subject in enumerate(subject_names):
            if subject in sample_name:
                subject_labelcnt[idx] += labelcnt
        cntsample += 1

    all_labelcnt = np.array(all_labelcnt)

    return subject_labelcnt, subject_names


def compute_label_frequency(label_path):
    # load label
    with open(label_path, 'rb') as f:
        sample_names, labels = pickle.load(f)
    f.close()
    labels = np.concatenate(labels, axis=0)
    all_labelcnt = np.sum(labels, axis=0)

    return all_labelcnt / labels.shape[0]


def compute_class_frequency(label_path):
    # load label
    with open(label_path, 'rb') as f:
        sample_names, labels = pickle.load(f)
    f.close()
    labels = np.concatenate(labels, axis=0)
    all_labelcnt = np.sum(labels, axis=0)

    return all_labelcnt / all_labelcnt.sum()


def split_data_random(args, infodir, kfold, splits_index):

    subject_labelcnt, subject_names = compute_frequency(args, infodir)
    subject_labelcnt = subject_labelcnt.tolist()
    idxs = list(zip(subject_labelcnt, subject_names))
    random.shuffle(idxs)
    subject_labelcnt[:], subject_names[:] = zip(*idxs)
    subject_labelcnt = np.array(subject_labelcnt)

    splits_labelcnt = np.zeros((kfold, subject_labelcnt.shape[1]))
    splits = -1 * np.ones((kfold, subject_labelcnt.shape[0]))
    cnts = np.zeros(kfold)

    for i in range(subject_labelcnt.shape[0]):
        kidx = i % kfold
        splits_labelcnt[kidx] += subject_labelcnt[i]
        splits[kidx][int(cnts[kidx])] = i
        cnts[kidx] += 1

    print("splits labelcnt:\n", splits_labelcnt)

    with open(
            os.path.join(args.datasetdir,
                         "splits_r" + str(splits_index) + ".txt"), 'w') as f:
        for i in range(kfold):
            cnt = 0
            while splits[i][cnt] != -1:
                f.write(subject_names[int(splits[i][cnt])] + ' ')
                cnt += 1
                if cnt == subject_labelcnt.shape[0]:
                    break
            f.write('\n')
    f.close()

    return