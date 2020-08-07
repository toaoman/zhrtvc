import numpy as np
# from scipy.io.wavfile import read
import torch
import librosa
import os

def read(fpath):
    wav, sr = librosa.load(fpath)
    out = np.clip(wav, -1, 1) * (2 ** 15 - 1)
    return sr, out.astype(int)


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))  # out=torch.cuda.LongTensor(max_len)
    mask = (ids < lengths.unsqueeze(1))#.bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    curdir = os.path.dirname(os.path.abspath(filename))
    filepaths_and_text = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split(split)
            if len(tmp) == 2:
                tmp.append('0')
            tmp[0] = os.path.join(curdir, tmp[0])
            filepaths_and_text.append(tmp)
    return filepaths_and_text


def load_filepaths_and_text_train(filename, split="|"):
    curdir = os.path.dirname(os.path.abspath(filename))
    filepaths_and_text = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split(split)
            tmp[0] = os.path.join(curdir, 'npy', tmp[0])
            filepaths_and_text.append(tmp)
    return filepaths_and_text



def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
