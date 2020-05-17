# zhrtvc
Chinese Real Time Voice Cloning

### 版本

v1.1.3

**变更**

- 从aukit.audio_io模块导入Dict2Obj。
- toolbox可视化显示合成的embed，alignment，spectrogram。
- toolbox录音修正格式不一致的bug。
- 增加代码行工具demo_cli。
- toolbox增加Preprocess的语音预处理按键，降噪和去除静音。
- 修正toolbox合成语音结尾截断的bug。

详见[readme](zhrtvc/README.md)


**toolbox**

![toolbox](https://github.com/KuangDD/zhrtvc/blob/master/files/toolbox.png)

### 目录介绍

#### zhrtvc
代码，包括encoder、synthesizer、vocoder、toolbox模块，包括模型训练的模块和可视化合成语音的模块。

执行脚本需要进入zhrtvc目录操作。

代码相关的说明详见zhrtvc目录下的[readme](zhrtvc/README.md)文件。


#### models
预训练的模型，包括encoder、synthesizer、vocoder的模型。

预训练的模型在百度网盘下载，下载后解压，替换models文件夹即可。

链接：https://pan.baidu.com/s/14hmJW7sY5PYYcCFAbqV0Kw 

提取码：zl9i


#### data
语料样例，包括语音和文本对齐语料，处理好的用于训练synthesizer的数据样例。

可以直接执行`synthesizer_preprocess_audio.py`和`synthesizer_preprocess_embeds.py`把samples的语音文本对齐语料转为SV2TTS的用于训练synthesizer的数据。

语料样例在百度网盘下载，下载后解压，替换data文件夹即可。

链接：https://pan.baidu.com/s/1Q_WUrmb7MW_6zQSPqhX9Vw 

提取码：bivr


### Real-Time Voice Cloning
This repository is an implementation of [Transfer Learning from Speaker Verification to
Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS) with a vocoder that works in real-time. Feel free to check [my thesis](https://matheo.uliege.be/handle/2268.2/6801) if you're curious or if you're looking for info I haven't documented yet (don't hesitate to make an issue for that too). Mostly I would recommend giving a quick look to the figures beyond the introduction.

SV2TTS is a three-stage deep learning framework that allows to create a numerical representation of a voice from a few seconds of audio, and to use it to condition a text-to-speech model trained to generalize to new voices.

### Papers implemented  
| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | This repo |
|[1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN (vocoder) | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|[1712.05884](https://arxiv.org/pdf/1712.05884.pdf) | Tacotron 2 (synthesizer) | Natural TTS Synthesis by Conditioning Wavenet on Mel Spectrogram Predictions | [Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2)
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E (encoder)| Generalized End-To-End Loss for Speaker Verification | This repo |
