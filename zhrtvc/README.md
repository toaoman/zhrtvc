# 中文语音克隆

## 使用指引
主要做synthesizer的部分，encoder和vocoder都用publish的模型。

### 训练

1. 处理语料，生成用于训练synthesizer的数据。

```markdown
python synthesizer_preprocess_audio.py
# 可带参数

usage: synthesizer_preprocess_audio.py [-h] [--datasets_root DATASETS_ROOT]
                                       [--datasets DATASETS] [-o OUT_DIR]
                                       [-n N_PROCESSES] [-s SKIP_EXISTING]
                                       [--hparams HPARAMS]

把语音信号转为频谱等模型训练需要的数据。

optional arguments:
  -h, --help            show this help message and exit
  --datasets_root DATASETS_ROOT
                        Path to the directory containing your datasets.
                        (default: ../data)
  --datasets DATASETS   Path to the directory containing your datasets.
                        (default: samples)
  -o OUT_DIR, --out_dir OUT_DIR
                        Path to the output directory that will contain the mel
                        spectrograms, the audios and the embeds. Defaults to
                        <datasets_root>/SV2TTS/synthesizer/
  -n N_PROCESSES, --n_processes N_PROCESSES
                        Number of processes in parallel. (default: 0)
  -s SKIP_EXISTING, --skip_existing SKIP_EXISTING
                        Whether to overwrite existing files with the same
                        name. Useful if the preprocessing was interrupted.
                        (default: True)
  --hparams HPARAMS     Hyperparameter overrides as a json string, for
                        example: '"key1":123,"key2":true' (default: )

```

```markdown
python synthesizer_preprocess_embeds.py
# 可带参数

usage: synthesizer_preprocess_embeds.py [-h]
                                        [--synthesizer_root SYNTHESIZER_ROOT]
                                        [-e ENCODER_MODEL_FPATH]
                                        [-n N_PROCESSES] [--hparams HPARAMS]

把语音信号转为语音表示向量。

optional arguments:
  -h, --help            show this help message and exit
  --synthesizer_root SYNTHESIZER_ROOT
                        Path to the synthesizer training data that contains
                        the audios and the train.txt file. If you let
                        everything as default, it should be
                        <datasets_root>/SV2TTS/synthesizer/. (default:
                        ../data/SV2TTS/synthesizer)
  -e ENCODER_MODEL_FPATH, --encoder_model_fpath ENCODER_MODEL_FPATH
                        Path your trained encoder model. (default:
                        ../models/encoder/saved_models/ge2e_pretrained.pt)
  -n N_PROCESSES, --n_processes N_PROCESSES
                        Number of parallel processes. An encoder is created
                        for each, so you may need to lower this value on GPUs
                        with low memory. Set it to 1 if CUDA is unhappy.
                        (default: 4)
  --hparams HPARAMS     Hyperparameter overrides as a json string, for
                        example: '"key1":123,"key2":true' (default: )

```

- **语料格式**

```markdown
|--datasets_root
   |--dataset1
      |--audio_dir1
      |--audio_dir2
      |--metadata.csv
   |--dataset2
```

- **metadata.csv**

一行描述一个音频文件。

每一行的数据格式：

```markdown
音频文件相对路径\t文本内容\n
```


- 例如：

```markdown
aishell/S0093/BAC009S0093W0368.mp3  有 着 对 美 和 品质 感 执着 的 追求
```

- 注意：

文本可以是汉字、拼音，汉字可以是分词后的汉字序列。

2. 训练模型，用处理好的数据训练synthesizer的模型。

```markdown
python synthesizer_train.py
# 可带参数

usage: synthesizer_train.py [-h] [--name NAME]
                            [--synthesizer_root SYNTHESIZER_ROOT]
                            [-m MODELS_DIR] [--mode MODE] [--GTA GTA]
                            [--restore RESTORE]
                            [--summary_interval SUMMARY_INTERVAL]
                            [--embedding_interval EMBEDDING_INTERVAL]
                            [--checkpoint_interval CHECKPOINT_INTERVAL]
                            [--eval_interval EVAL_INTERVAL]
                            [--tacotron_train_steps TACOTRON_TRAIN_STEPS]
                            [--tf_log_level TF_LOG_LEVEL]
                            [--slack_url SLACK_URL] [--hparams HPARAMS]

训练语音合成器模型。

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Name of the run and of the logging directory.
                        (default: synz)
  --synthesizer_root SYNTHESIZER_ROOT
                        Path to the synthesizer training data that contains
                        the audios and the train.txt file. If you let
                        everything as default, it should be
                        <datasets_root>/SV2TTS/synthesizer/. (default:
                        ../data/SV2TTS/synthesizer)
  -m MODELS_DIR, --models_dir MODELS_DIR
                        Path to the output directory that will contain the
                        saved model weights and the logs. (default:
                        ../models/synthesizer/saved_models/)
  --mode MODE           mode for synthesis of tacotron after training
                        (default: synthesis)
  --GTA GTA             Ground truth aligned synthesis, defaults to True, only
                        considered in Tacotron synthesis mode (default: True)
  --restore RESTORE     Set this to False to do a fresh training (default:
                        True)
  --summary_interval SUMMARY_INTERVAL
                        Steps between running summary ops (default: 100)
  --embedding_interval EMBEDDING_INTERVAL
                        Steps between updating embeddings projection
                        visualization (default: 100)
  --checkpoint_interval CHECKPOINT_INTERVAL
                        Steps between writing checkpoints (default: 1000)
  --eval_interval EVAL_INTERVAL
                        Steps between eval on test data (default: 100)
  --tacotron_train_steps TACOTRON_TRAIN_STEPS
                        total number of tacotron training steps (default:
                        500000)
  --tf_log_level TF_LOG_LEVEL
                        Tensorflow C++ log level. (default: 1)
  --slack_url SLACK_URL
                        slack webhook notification destination link (default:
                        None)
  --hparams HPARAMS     Hyperparameter overrides as a json string, for
                        example: '"key1":123,"key2":true' (default: )

```

- **语料格式**

```markdown
|--synthesizer
   |--embeds
   |--mels
   |--train.txt
```

- **train.txt**

一行描述一个训练样本。

每一行的数据格式：

```markdown
音频文件路径|mel文件路径|embed文件路径|音频帧数|mel帧数|文本
```

- 例如：
```markdown
../data/samples/aishell/S0093/BAC009S0093W0368.mp3|mel-aishell-S0093-BAC009S0093W0368.mp3.npy|embed-aishell-S0093-BAC009S0093W0368.mp3.npy|54656|216|有 着 对 美 和 品质 感 执着 的 追求
```

- 注意

mel文件路径和embed文件路径可以是相对路径（相对于train.txt所在文件夹），也可以是绝对路径。

如果多个数据一起用，可以用绝对路径表示，汇总到一个train.txt文件，便于训练。


## 版本记录

### v1.1.7
- 修改说明文档。
- 修正已知BUG。
- 增加实验的Mellotron的语音合成器模型。

### v1.1.5
- 修正phkit依赖版本错误。
- 提供项目的依赖及参考版本。
- 提供用开源数据训练的模型。
- 提供降噪和去除静音的预处理后的开源语料。

### v1.1.4
- Update train melgan. Fix some bugs.
- Update toolbox. Load synthesizer with hparams.
- Add tools for joint audios to train.

### v1.1.3
- 从aukit.audio_io模块导入Dict2Obj。
- toolbox可视化显示合成的embed，alignment，spectrogram。
- toolbox录音修正格式不一致的bug。
- 增加代码行工具demo_cli。
- toolbox增加Preprocess的语音预处理按键，降噪和去除静音。
- 修正toolbox合成语音结尾截断的bug。
- 样例文本提供长句和短句。
- 增加合成参考音频文本的按键Compare，对比参考语音和合成语音。


### v1.1.2
- 语音和频谱的处理使用工具包：aukit，用pip install aukit即可。
- 文本和音素的处理使用工具包：phkit，用pip install phkit即可。
- 提供预训练好的encoder、synthesizer、vocoder模型和语音样例。
- 工具盒toolbox界面的Dataset的Random按钮是随机选择文本，而非选择数据集。选择数据集需要手动下拉框选择。
- 预训练的synthesizer模型用ali句子的dataset训练的，用alijuzi的dataset的语音做参考音频效果较好。
- 重整模型和数据的目录结构，提供可训练的样例。


## 参考项目

- **Real-Time Voice Cloning**
This repository is an implementation of [Transfer Learning from Speaker Verification to
Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS) with a vocoder that works in real-time. Feel free to check [my thesis](https://matheo.uliege.be/handle/2268.2/6801) if you're curious or if you're looking for info I haven't documented yet (don't hesitate to make an issue for that too). Mostly I would recommend giving a quick look to the figures beyond the introduction.

SV2TTS is a three-stage deep learning framework that allows to create a numerical representation of a voice from a few seconds of audio, and to use it to condition a text-to-speech model trained to generalize to new voices.

**Video demonstration** (click the picture):

[![Toolbox demo](https://i.imgur.com/Ixy13b7.png)](https://www.youtube.com/watch?v=-O_hYhToKoA)



### Papers implemented  
| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | This repo |
|[1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN (vocoder) | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|[1712.05884](https://arxiv.org/pdf/1712.05884.pdf) | Tacotron 2 (synthesizer) | Natural TTS Synthesis by Conditioning Wavenet on Mel Spectrogram Predictions | [Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2)
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E (encoder)| Generalized End-To-End Loss for Speaker Verification | This repo |


## Quick start
### Requirements
You will need the following whether you plan to use the toolbox only or to retrain the models.

**Python 3.7**. Python 3.6 might work too, but I wouldn't go lower because I make extensive use of pathlib.

Run `pip install -r requirements.txt` to install the necessary packages. Additionally you will need [PyTorch](https://pytorch.org/get-started/locally/).

A GPU is mandatory, but you don't necessarily need a high tier GPU if you only want to use the toolbox.

### Pretrained models
Download the latest [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models).

### Preliminary
Before you download any dataset, you can begin by testing your configuration with:

`python demo_cli.py`

If all tests pass, you're good to go.

### Datasets
For playing with the toolbox alone, I only recommend downloading [`LibriSpeech/train-clean-100`](http://www.openslr.org/resources/12/train-clean-100.tar.gz). Extract the contents as `<datasets_root>/LibriSpeech/train-clean-100` where `<datasets_root>` is a directory of your choosing. Other datasets are supported in the toolbox, see [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Training#datasets). You're free not to download any dataset, but then you will need your own data as audio files or you will have to record it with the toolbox.

### Toolbox
You can then try the toolbox:

`python demo_toolbox.py -d <datasets_root>`  
or  
`python demo_toolbox.py`  

depending on whether you downloaded any datasets. If you are running an X-server or if you have the error `Aborted (core dumped)`, see [this issue](https://github.com/CorentinJ/Real-Time-Voice-Cloning/issues/11#issuecomment-504733590).

## Wiki
- **How it all works** (coming soon!)
- [**Training models yourself**](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Training)
- **Training with other data/languages** (coming soon! - see [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/issues/30#issuecomment-507864097) for now)
- [**TODO and planned features**](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/TODO-&-planned-features) 

## Contribution
Feel free to open issues or PRs for any problem you may encounter, typos that you see or aspects that are confusing. Contributions are welcome, open an issue or email me if you have something you want to work on.
