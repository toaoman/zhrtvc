#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/4/13
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
import librosa


def gen_filelists():
    indir = Path(r"E:\data\librispeech\LibriSpeech\test-clean\LibriSpeech\test-clean")
    # 61-70968-0001: speaker-book-id
    # 61-70968-0001 GIVE NOT SO EARNEST A MIND TO THESE MUMMERIES CHILD
    outs = []
    for fpath in indir.glob("**/*.txt"):
        for line in open(fpath):
            idx, *ws = line.strip().split()
            outpath = str(fpath.parent.joinpath(f"{idx}.flac")).replace("\\", "/")
            outtext = (" ".join(ws)).lower()
            outspeaker = idx.split("-")[0]
            outs.append([outpath, outtext, outspeaker])

    np.random.shuffle(outs)

    outpath = r"E:\data\librispeech\LibriSpeech\test-clean\lstc_total_filelist.txt"
    with open(outpath, "wt", encoding="utf8") as fout:
        for line in tqdm(outs):
            if 10 < len(line[1]) < 50:
                wav, sr = librosa.load(line[0])
                wav = (wav * (2 ** 15)).astype(int)
                if max(wav) >= 2 ** 15:
                    print(line)
                    continue
                fout.write(("|".join(line)) + "\n")




def read(fpath):
    wav, sr = librosa.load(fpath)
    out = wav * (2 ** 15 - 1)
    return sr, out.astype(int)


def load_flac():
    from scipy.io import wavfile
    import librosa
    inpath = r"E:/data/librispeech/LibriSpeech/test-clean/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"
    inpath = r"E:\data\record\kdd\wavs\kdd-4253.wav"
    inpath = r"E:/data/librispeech/LibriSpeech/test-clean/LibriSpeech/test-clean/2094/142345/2094-142345-0025.flac"
    # sr, wav = wavfile.read(inpath)
    wav2, sr = librosa.load(inpath)
    print(max(abs(wav2)))


if __name__ == "__main__":
    print(__file__)
    # gen_filelists()
    load_flac()
