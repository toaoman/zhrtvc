#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/4/14
"""
"""
from pathlib import Path
from functools import partial
from multiprocessing.pool import Pool
from matplotlib import pyplot as plt
from tqdm import tqdm
import collections as clt
import os
import re
import json
import numpy as np
import shutil

from data_utils import TextMelLoader
from hparams import create_hparams

hp = create_hparams()

metadata_path = None
text_mel_loader = None
output_dir = None


def format_index(index):
    return '{:06d}'.format(index)


def process_one(index):
    global text_mel_loader
    global metadata_path
    global output_dir
    if text_mel_loader is None:
        text_mel_loader = TextMelLoader(metadata_path, hparams=hp, mode='preprocess')
    text, mel, speaker_id, f0 = text_mel_loader[index]
    onedir = output_dir.joinpath(format_index(index))
    onedir.mkdir(exist_ok=True)
    tpath = onedir.joinpath("text.npy")
    mpath = onedir.joinpath("mel.npy")
    spath = onedir.joinpath("speaker.npy")
    fpath = onedir.joinpath("f0.npy")
    np.save(tpath, text.numpy(), allow_pickle=False)
    np.save(mpath, mel.numpy(), allow_pickle=False)
    np.save(spath, speaker_id.numpy(), allow_pickle=False)
    np.save(fpath, f0.numpy(), allow_pickle=False)
    return index


def process_many(n_processes):
    # Embed the utterances in separate threads
    ids = list(range(len(text_mel_loader)))
    with open(output_dir.parent.joinpath('train.txt'), 'wt', encoding='utf8') as fout:
        for idx in tqdm(ids):
            tmp = text_mel_loader.audiopaths_and_text[idx]
            fout.write('{}\t{}\n'.format(format_index(idx), '\t'.join(tmp).strip()))

    if n_processes == 0:
        for num in tqdm(ids):
            process_one(num)
    else:
        func = partial(process_one)
        job = Pool(n_processes).imap(func, ids)
        list(tqdm(job, "Embedding", len(ids), unit="utterances"))


if __name__ == "__main__":
    print(__file__)
    import argparse

    parser = argparse.ArgumentParser(
        description="预处理训练数据，保存为numpy的npy格式，训练的时候直接从本地load数据。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--metadata_path", type=str, default=r'F:\github\zhrtvc\data\samples\metadata.csv',
                        help="metadata file path")
    parser.add_argument("-o", "--output_dir", type=Path, default=Path(r'F:\github\zhrtvc\data\SV2TTS\mellotron\npy'),
                        help="Path to the output directory")
    parser.add_argument("-n", "--n_processes", type=int, default=0,
                        help="Number of processes in parallel.")
    parser.add_argument("-s", "--skip_existing", type=bool, default=True,
                        help="Whether to overwrite existing files with the same name. ")
    parser.add_argument("--hparams", type=str, default="",
                        help="Hyperparameter overrides as a comma-separated list of name-value pairs")
    args = parser.parse_args()

    # Process the arguments
    if not hasattr(args, "output_dir"):
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "mellotron")

    metadata_path = args.metadata_path
    text_mel_loader = TextMelLoader(metadata_path, hparams=hp, mode='preprocess')

    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    # Preprocess the dataset
    process_many(args.n_processes)
