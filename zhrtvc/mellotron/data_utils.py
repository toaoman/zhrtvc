from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__name__).stem)

import random
import os
import re
import numpy as np
import torch
import torch.utils.data
import librosa

import layers
from pathlib import Path
from utils import load_wav_to_torch, load_filepaths_and_text, load_filepaths_and_text_train
from text import text_to_sequence, cmudict
from yin import compute_yin

from utils import melspectrogram_torch, linearspectrogram_torch


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text and speaker ids
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms and f0s from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams, speaker_ids=None, mode='train'):
        tmp = mode.split('-')
        if tmp[0] == 'train':
            self.audiopaths_and_text = load_filepaths_and_text_train(audiopaths_and_text, split='\t')
            if len(tmp) == 2:
                self.mode = tmp[1]
            else:
                self.mode = True
        else:
            self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text, split='\t')
            self.mode = False
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.f0_min = hparams.f0_min
        self.f0_max = hparams.f0_max
        self.harm_thresh = hparams.harm_thresh
        self.p_arpabet = hparams.p_arpabet

        self.cmudict = None
        if hparams.cmudict_path is not None:
            self.cmudict = cmudict.CMUDict(hparams.cmudict_path)

        self.speaker_ids = speaker_ids

        if self.speaker_ids is None:
            self.speaker_ids = self.create_speaker_lookup_table(self.audiopaths_and_text)

        # random.seed(1234)
        # random.shuffle(self.audiopaths_and_text)

    def get_data_train(self, data_dir):
        onedir = Path(data_dir)
        tpath = onedir.joinpath("text.npy")
        mpath = onedir.joinpath("mel.npy")
        spath = onedir.joinpath("speaker.npy")
        fpath = onedir.joinpath("f0.npy")
        text = torch.from_numpy(np.load(tpath))  # (86,)
        mel = torch.from_numpy(np.load(mpath))  # (80, 397)
        speaker_id = torch.from_numpy(np.load(spath))  # (1,)
        f0 = np.load(fpath)  # (1, 395)
        if self.mode == 'f01':
            # 用f0数据。
            f0 = f0[:, :mel.shape[1]]
        elif self.mode == 'f02':
            # 用f0的均值代替f0，简化f0。
            f0 = f0.flatten()
            f0_value = np.mean(f0[f0 > 10])
            f0 = np.ones((1, mel.shape[1])) * f0_value
        elif self.mode == 'f03':
            # 用零向量填充f0。
            f0 = np.zeros((1, mel.shape[1]))
        elif self.mode == 'f04':
            # 不用f0。
            f0 = None
        else:
            # 默认：不用f0。
            f0 = None
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0)
        return (text, mel, speaker_id, f0)

    def create_speaker_lookup_table(self, audiopaths_and_text):
        speaker_ids = np.sort(np.unique([x[-1] if len(x) >= 3 else '0' for x in audiopaths_and_text]))
        d = {speaker_ids[i]: i for i in range(len(speaker_ids))}
        return d

    def get_f0(self, audio, sampling_rate=22050, frame_length=1024,
               hop_length=256, f0_min=100, f0_max=300, harm_thresh=0.1):
        f0, harmonic_rates, argmins, times = compute_yin(
            audio, sampling_rate, frame_length, hop_length, f0_min, f0_max,
            harm_thresh)
        pad = int((frame_length / hop_length) / 2)
        f0 = [0.0] * pad + f0 + [0.0] * pad

        f0 = np.array(f0, dtype=np.float32)
        return f0

    def get_data(self, audiopath_and_text):
        audiopath, text, speaker = audiopath_and_text
        text = self.get_text(text)
        mel, f0 = self.get_mel_and_f0(audiopath)
        speaker_id = self.get_speaker_id(speaker)
        return (text, mel, speaker_id, f0)

    def get_speaker_id(self, speaker_id):
        return torch.IntTensor([self.speaker_ids[speaker_id]])

    def get_mel_and_f0(self, filepath):
        audio, sampling_rate = load_wav_to_torch(filepath)
        audio_norm = audio / self.max_wav_value
        # if sampling_rate != self.stft.sampling_rate:
        #     raise ValueError("{} SR doesn't match target {} SR".format(
        #         sampling_rate, self.stft.sampling_rate))
        # audio_norm = audio_norm.unsqueeze(0)
        # melspec = self.stft.mel_spectrogram(audio_norm)
        # melspec = torch.squeeze(melspec, 0)

        melspec = linearspectrogram_torch(audio_norm)  # 用aukit的频谱生成方案

        f0 = self.get_f0(audio.cpu().numpy(), sampling_rate,
                         self.filter_length, self.hop_length, self.f0_min,
                         self.f0_max, self.harm_thresh)
        f0 = torch.from_numpy(f0)[None]
        # f0 = f0[:, :melspec.size(1)]

        # 用零向量替换F0
        # f0 = torch.zeros(1, melspec.shape[1], dtype=torch.float)
        return melspec, f0

    def get_text(self, text):
        text_norm = torch.IntTensor(
            text_to_sequence(text, self.text_cleaners))  # self.cmudict, self.p_arpabet))

        return text_norm

    def __getitem__(self, index):
        if self.mode:
            tmp = index
            while True:
                try:  # 模型训练模式容错。
                    out = self.get_data_train(self.audiopaths_and_text[tmp][0])
                    if tmp != index:
                        logger.info(
                            'The index <{}> loaded success!\n{}\n'.format(tmp, '-' * 50))
                    return out
                except:
                    logger.info(
                        'The index <{}> loaded failed!'.format(index, tmp))
                    tmp = np.random.randint(0, len(self.audiopaths_and_text) - 1)
        else:
            try:  # 数据预处理模式容错。
                out = self.get_data(self.audiopaths_and_text[index])
                return out
            except Exception as e:
                logger.info('The index <{}> loaded failed!'.format(index))
                return

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        speaker_ids = torch.LongTensor(len(batch))
        f0_padded = torch.FloatTensor(len(batch), 1, max_target_len)
        f0_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][2]
            f0 = batch[ids_sorted_decreasing[i]][3]
            if isinstance(f0, torch.Tensor):
                f0_padded[i, :, :f0.size(1)] = f0
            else:
                f0_padded = f0

        model_inputs = (text_padded, input_lengths, mel_padded, gate_padded,
                        output_lengths, speaker_ids, f0_padded)

        return model_inputs
