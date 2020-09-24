#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/18
"""
"""
from phkit.chinese import text_to_sequence as text_to_sequence_phkit, sequence_to_text
from .parse_ssml import convert_ssml


def text_to_sequence(text, cleaner_names, **kwargs):
    """
    文本转为向量。
    """
    if cleaner_names == 'ssml':
        zp_lst = convert_ssml(text)
        pin_text = ' '.join([p for z, p in zp_lst])
        seq = text_to_sequence_phkit(pin_text, cleaner_names='pinyin')
    else:
        seq = text_to_sequence_phkit(text, cleaner_names=cleaner_names)
    return seq


if __name__ == "__main__":
    print(__file__)
    pinyin_text = "ka3 er3 pu3 pei2 wai4 sun1 wan2 hua2 ti1 . "
    ssml_text = '<speak><phoneme alphabet="py" ph="gan4 ma2 a5 ni3">干嘛啊你</phoneme>？？<phoneme alphabet="py" ph="you4 lai2">又来</phoneme><phoneme alphabet="py" ph="gou1 da5 shei2">勾搭谁</phoneme>。</speak>'
    hanzi_text = '你好。'

    out = text_to_sequence(pinyin_text, cleaner_names='pinyin')
    print(sequence_to_text(out))
    # k a 3 - ee er 2 - p u 3 - p ei 2 - uu uai 4 - s un 1 - uu uan 2 - h ua 2 - t i 1 - . - ~ _

    out = text_to_sequence(ssml_text, cleaner_names='ssml')
    print(sequence_to_text(out))
    # g an 4 - m a 2 - aa a 5 - n i 3 - ? - ? - ii iu 4 - l ai 2 - g ou 1 - d a 5 - sh ei 2 - . - ~ _

    out = text_to_sequence(hanzi_text, cleaner_names='hanzi')
    print(sequence_to_text(out))
    # n i 2 - h ao 3 - . - ~ _
