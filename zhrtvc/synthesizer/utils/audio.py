from aukit.audio_griffinlim import load_wav, save_wav, save_wavenet_wav, preemphasis, inv_preemphasis
from aukit.audio_griffinlim import start_and_end_indices, get_hop_size
from aukit.audio_griffinlim import linear_spectrogram as linearspectrogram
from aukit.audio_griffinlim import mel_spectrogram
from aukit.audio_griffinlim import inv_linear_spectrogram, inv_mel_spectrogram
from aukit.audio_griffinlim import librosa_pad_lr
from aukit.audio_griffinlim import default_hparams, Dict2Obj
import numpy as np

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


def melspectrogram(wav, hparams=None):
    wav = np.pad(wav.flatten(), (_pad_len, _pad_len), mode="reflect")
    mel = mel_spectrogram(wav, melgan_hparams)
    mel = mel / 20
    return mel


def inv_melspectrogram(mel, hparams=None):
    mel = mel * 20
    wav = inv_mel_spectrogram(mel, melgan_hparams)
    return wav


if __name__ == "__main__":
    import aukit

    inpath = r"E:\data\temp\01.wav"
    wav = load_wav(inpath, sr=16000)
    mel = melspectrogram(wav)
    out = inv_melspectrogram(mel)
    aukit.play_audio(wav)
    aukit.play_audio(out)
