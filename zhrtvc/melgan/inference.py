#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/20
"""
"""
from melgan.mel2wav import MelVocoder

from pathlib import Path
from tqdm import tqdm
from scipy.io import wavfile
import argparse
import librosa
import torch
import numpy as np
import traceback

from melgan.train import audio2mel, audio2mel_mellotron, audio2mel_synthesizer


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
                        default=Path("../../models/vocoder/saved_models/melgan/melgan_multi_speaker.pt"))
    parser.add_argument("-o", "--save_path", type=Path, default=Path("../../data/temp/melgan"))
    parser.add_argument("-i", "--folder", type=Path, default=Path("../../data/samples/biaobei/biaobei"))
    parser.add_argument("--mode", type=str, default='synthesizer')
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    vocoder = MelVocoder(args.load_path, github=True, model_name=Path(args.load_path).stem, device="cuda")
    args.save_path.mkdir(exist_ok=True, parents=True)

    if args.mode == 'default':
        fft = audio2mel
    elif args.mode == 'synthesizer':
        fft = audio2mel_synthesizer
    elif args.mode == 'mellotron':
        fft = audio2mel_mellotron
    else:
        raise KeyError
    fpath_lst = list(args.folder.glob("**/*"))
    fpath_choices = np.random.choice(fpath_lst, min(args.n_samples, len(fpath_lst)), replace=False)
    for i, fname in enumerate(tqdm(fpath_choices)):
        try:
            wav, sr = librosa.core.load(str(fname))
            mel = fft(torch.from_numpy(wav[None]))
            recons = vocoder.inverse(mel).squeeze().cpu().numpy()
            wavfile.write(filename=str(args.save_path.joinpath(f'{fname.stem}_raw.wav')), rate=sr, data=wav)
            wavfile.write(filename=str(args.save_path.joinpath(f'{fname.stem}_syn.wav')), rate=sr, data=recons)
            # librosa.output.write_wav(args.save_path.joinpath(f'{fname.stem}.wav'), recons, sr=sr)
        except:
            traceback.print_exc()


if __name__ == "__main__":
    main()
