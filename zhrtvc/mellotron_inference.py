#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/11/28
"""
mellotron_inference
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__name__).stem)

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--checkpoint_path', type=str,
                        default=r"../models/mellotron/samples/checkpoint-000000.pt",
                        help='模型路径。')
    parser.add_argument('-s', '--speakers_path', type=str,
                        default=r"../models/mellotron/samples/metadata/speakers.json",
                        help='发音人映射表路径。')
    parser.add_argument("-o", "--out_dir", type=Path, default=r"../models/mellotron/samples/test",
                        help='保存合成的数据路径。')
    parser.add_argument("-p", "--play", type=int, default=0,
                        help='是否合成语音后自动播放语音。')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--hparams_path', type=str,
                        default=r"../models/mellotron/samples/metadata/hparams.json",
                        required=False, help='comma separated name=value pairs')
    parser.add_argument("-e", "--encoder_model_fpath", type=Path,
                        default=r"../models/encoder/saved_models/ge2e_pretrained.pt",
                        help="Path your trained encoder model.")
    parser.add_argument("--save_model_path", type=str, default='',
                        help='保存模型为可以直接torch.load的格式')
    parser.add_argument("--cuda", type=str, default='-1',
                        help='设置CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    return args


args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

import aukit
import time
import json
import traceback
import torch
import numpy as np

from mellotron.inference import create_hparams, load_model_mellotron, MellotronSynthesizer, griffinlim_vocoder
from mellotron.inference import save_model
from utils.texthelper import xinqing_texts

aliaudio_fpaths = [str(w) for w in sorted(Path(r'../data/samples/aliaudio').glob('*/*.mp3'))]

if __name__ == "__main__":
    args_hparams = open(args.hparams_path, encoding='utf8').read()
    _hparams = create_hparams(args_hparams)

    model_path = args.checkpoint_path
    load_model_mellotron(model_path, hparams=_hparams)

    ## Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
              "%.1fGb total memory.\n" %
              (torch.cuda.device_count(),
               device_id,
               gpu_properties.name,
               gpu_properties.major,
               gpu_properties.minor,
               gpu_properties.total_memory / 1e9))

    msyner = MellotronSynthesizer(model_path=args.checkpoint_path, speakers_path=args.speakers_path,
                                  hparams_path=args.hparams_path)

    if args.save_model_path:
        save_model(msyner, args.save_model_path)

    spec = msyner.synthesize(text='你好，欢迎使用语言合成服务。', speaker='speaker')

    ## Run a test
    # spec, align = synthesize_one('你好，欢迎使用语言合成服务。', aliaudio_fpaths[0], with_alignment=True,
    #                              hparams=_hparams, encoder_fpath=args.encoder_model_fpath)
    print("Spectrogram shape: {}".format(spec.shape))
    # print("Alignment shape: {}".format(align.shape))
    wav = griffinlim_vocoder(spec)
    print("Waveform shape: {}".format(wav.shape))

    print("All test passed! You can now synthesize speech.\n\n")

    print("Interactive generation loop")
    num_generated = 0
    args.out_dir.mkdir(exist_ok=True, parents=True)
    speaker_index_dict = json.load(open(args.speakers_path, encoding='utf8'))
    speaker_names = list(speaker_index_dict.keys())
    example_texts = xinqing_texts
    example_fpaths = aliaudio_fpaths
    while True:
        try:
            # Get the reference audio filepath
            speaker = input("Speaker:\n")
            if not speaker.strip():
                speaker = np.random.choice(speaker_names)
            print('Speaker: {}'.format(speaker))

            ## Generating the spectrogram
            text = input("Text:\n")
            if not text.strip():
                text = np.random.choice(example_texts)
            print('Text: {}'.format(text))
            # The synthesizer works in batch, so you need to put your data in a list or numpy array

            print("Creating the spectrogram ...")
            spec = msyner.synthesize(text=text, speaker=speaker)
            # spec, align = synthesize_one(text, speaker=speaker, with_alignment=True,
            #                              hparams=_hparams, encoder_fpath=args.encoder_model_fpath)
            print("Spectrogram shape: {}".format(spec.shape))
            # print("Alignment shape: {}".format(align.shape))
            ## Generating the waveform
            print("Synthesizing the waveform ...")
            wav = griffinlim_vocoder(spec)
            print("Waveform shape: {}".format(wav.shape))

            # Save it on the disk
            cur_time = time.strftime('%Y%m%d_%H%M%S')
            fpath = args.out_dir.joinpath("demo_out_{}.wav".format(cur_time))
            # librosa.output.write_wav(fpath, generated_wav.astype(np.float32), synthesizer.sample_rate)
            aukit.save_wav(wav, fpath, sr=_hparams.sampling_rate)  # save

            txt_path = args.out_dir.joinpath("info_dict.txt".format(cur_time))
            with open(txt_path, 'at', encoding='utf8') as fout:
                dt = dict(text=text, audio_path=str(fpath), speaker=speaker, time=cur_time)
                out = json.dumps(dt, ensure_ascii=False)
                fout.write('{}\n'.format(out))

            num_generated += 1
            print("\nSaved output as %s\n\n" % fpath)
            if args.play:
                aukit.play_audio(fpath, sr=_hparams.sampling_rate)
        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
            traceback.print_exc()
