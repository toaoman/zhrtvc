# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/9/22
"""
synthesize
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__name__).stem)

import torch

from model import Tacotron2, load_model
from text import cmudict, text_to_sequence
from utils import inv_linear_spectrogram

from hparams import create_hparams

_hparams = create_hparams()
_mellotron = None
_device = 'cpu'


def load_model_mellotron(model_path):
    global _mellotron
    _mellotron = load_model(_hparams).to(_device).eval()
    _mellotron.load_state_dict(torch.load(model_path)['state_dict'])


def synthesize_one(text, speaker=0, model_path=''):
    if _mellotron is None:
        load_model_mellotron(model_path)

    text_encoded = torch.LongTensor(text_to_sequence(text, _hparams.text_cleaners))[None, :].to(_device)
    speaker_id = torch.LongTensor([speaker]).to(_device)
    style_input = 0
    pitch_contour = torch.zeros(1, 1, text_encoded.shape[1] * 5, dtype=torch.float)
    pitch_contour = None

    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = _mellotron.inference(
            (text_encoded, style_input, speaker_id, pitch_contour))

    out_mel = mel_outputs_postnet.data.cpu().numpy()[0]
    return out_mel


def griffinlim_vocoder(spec):
    wav = inv_linear_spectrogram(spec)
    return wav


if __name__ == "__main__":
    print(__file__)
    model_path = r'F:\github\zhrtvc\models\mellotron\linear\checkpoint-000000.pt'
    load_model_mellotron(model_path)

    spec = synthesize_one('你好。', 0)
    print(spec.shape)
    wav = griffinlim_vocoder(spec)
    print(wav.shape)
