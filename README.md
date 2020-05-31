![zhrtvc](zhrtvc.png "zhrtvc")

# zhrtvc
Zhongwen Real Time Voice Cloning

### 版本

v1.1.4

详见[readme](zhrtvc/README.md)


* **原始语音和克隆语音对比样例**

链接: https://pan.baidu.com/s/1TQwgzEIxD2VBrVZKCblN1g 

提取码: 8ucd


* **toolbox**

![toolbox](files/toolbox.png "toolbox")


* **合成样例**

[aliaudio-Aibao-004113.wav](files/aliaudio-Aibao-004113.wav)

[aliaudio-Aimei-007261.wav](files/aliaudio-Aimei-007261.wav)

[aliaudio-Aina-000819.wav](files/aliaudio-Aina-000819.wav)

[aliaudio-Aiqi-009619.wav](files/aliaudio-Aiqi-009619.wav)

[aliaudio-Aitong-003149.wav](files/aliaudio-Aitong-003149.wav)

[aliaudio-Aiwei-009461.wav](files/aliaudio-Aiwei-009461.wav)


* **注意**

跑提供的模型建议用Griffin-Lim声码器，目前MelGAN和WaveRNN没有完全适配。


### 目录介绍

#### zhrtvc
代码，包括encoder、synthesizer、vocoder、toolbox模块，包括模型训练的模块和可视化合成语音的模块。

执行脚本需要进入zhrtvc目录操作。

代码相关的说明详见zhrtvc目录下的[readme](zhrtvc/README.md)文件。


#### models
预训练的模型，包括encoder、synthesizer、vocoder的模型。

预训练的模型在百度网盘下载，下载后解压，替换models文件夹即可。

* **样本模型**

链接：https://pan.baidu.com/s/14hmJW7sY5PYYcCFAbqV0Kw 

提取码：zl9i


#### data
语料样例，包括语音和文本对齐语料，处理好的用于训练synthesizer的数据样例。

可以直接执行`synthesizer_preprocess_audio.py`和`synthesizer_preprocess_embeds.py`把samples的语音文本对齐语料转为SV2TTS的用于训练synthesizer的数据。

语料样例在百度网盘下载，下载后解压，替换data文件夹即可。

* **样本数据**

链接：https://pan.baidu.com/s/1Q_WUrmb7MW_6zQSPqhX9Vw 

提取码：bivr


### 学习交流

【AI解决方案交流群】QQ群：925294583

点击链接加入群聊：https://jq.qq.com/?_wv=1027&k=wlQzvT0N


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
