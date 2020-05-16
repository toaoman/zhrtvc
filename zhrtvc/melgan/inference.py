#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/20
"""
"""
from melgan.mel2wav import MelVocoder

from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import torch
import numpy as np

from aukit.audio_griffinlim import mel_spectrogram, default_hparams
from aukit.audio_io import Dict2Obj
import aukit

_melgan_vocoder = None

_sr = 22050
my_hp = {
    "n_fft": 1024, "hop_size": 256, "win_size": 1024,
    "sample_rate": _sr,
    "fmin": 0, "fmax": _sr // 2,
    "preemphasize": False,
    'symmetric_mels': True,
    'signal_normalization': False,
    'allow_clipping_in_normalization': False,
    'ref_level_db': 0,
    '__file__': __file__
}

melgan_hparams = {}
melgan_hparams.update(default_hparams)
melgan_hparams.update(my_hp)
melgan_hparams = Dict2Obj(melgan_hparams)

_pad_len = (default_hparams.n_fft - default_hparams.hop_size) // 2


def wav2mel(wav, hparams=None):
    # mel = Audio2Mel().cuda()(src)
    # return mel
    hparams = hparams or melgan_hparams
    wav = np.pad(wav.flatten(), (_pad_len, _pad_len), mode="reflect")
    mel = mel_spectrogram(wav, hparams)
    mel = mel / 20
    return mel


def load_vocoder_melgan(load_path):
    global _melgan_vocoder
    if _melgan_vocoder is None:
        _melgan_vocoder = MelVocoder(load_path, github=True, model_name=Path(load_path).stem)
    return _melgan_vocoder


def infer_waveform_melgan(mel, load_path=None):
    global _melgan_vocoder
    if _melgan_vocoder is None:
        _melgan_vocoder = MelVocoder(load_path, github=True, model_name=Path(load_path).stem)

    mel = torch.from_numpy(mel[np.newaxis].astype(np.float32))
    wav = _melgan_vocoder.inverse(mel).squeeze().cpu().numpy()
    return wav


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=Path,
                        default=Path(r"../vocoder/saved_models/melgan/multi_speaker.pt"))
    parser.add_argument("--save_path", type=Path,
                        default=Path(r"E:\lab\melgan\data\melgan\aliexamples_mel_multi_22050_pad"))
    parser.add_argument("--folder", type=Path,
                        default=Path(r"E:\lab\melgan\data\aliexamples"))
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    vocoder = MelVocoder(args.load_path, github=True, model_name=Path(args.load_path).stem, device="cuda")
    args.save_path.mkdir(exist_ok=True, parents=True)

    for i, fname in tqdm(enumerate(args.folder.glob("*.wav"))):
        wavname = fname.stem
        wav, sr = librosa.core.load(fname)
        mel = vocoder(torch.from_numpy(wav)[None])
        recons = vocoder.inverse(mel).squeeze().cpu().numpy()
        librosa.output.write_wav(args.save_path / (wavname + ".wav"), recons, sr=22050)


def run_compare():
    args = parse_args()
    load_vocoder_melgan(args.load_path)
    for i, fname in tqdm(enumerate(args.folder.glob("*.wav"))):
        wav, sr = librosa.core.load(fname, sr=16000)
        mel = wav2mel(wav)
        out = infer_waveform_melgan(mel=mel)
        aukit.play_audio(wav, sr=sr)
        aukit.play_audio(out, sr=sr)


if __name__ == "__main__":
    # main()
    run_compare()
