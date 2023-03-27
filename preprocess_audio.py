#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from os.path import join as path_join, exists
import numpy as np
from scipy.io import wavfile
import scipy.signal as si
import matplotlib.pyplot as plt
from pydub import AudioSegment
from utils import extract_wav, norm_wav
import soundfile as sf
import pyloudnorm as pyln
from pyloudnorm import IIRfilter
from pydub.utils import make_chunks


def normalize_audio(input_file, output_file):
    # Load the input audio file
    audio = AudioSegment.from_file(input_file)

    # Set the maximum target amplitude to -20 dBFS
    target_dBFS = -20

    # Calculate the current volume and the necessary change in volume
    current_dBFS = audio.dBFS
    dBFS_change = target_dBFS - current_dBFS

    # Calculate the ratio of the necessary volume change
    volume_ratio = 10 ** (dBFS_change / 20)

    # Get the average volume of each 1-second segment
    segment_volumes = []
    segments = []
    for chunk in make_chunks(audio, 1000):
        segment_volumes.append(chunk.dBFS)

    start = 0
    for i in range(len(segment_volumes)):
        if abs(segment_volumes[i] - segment_volumes[start]) >= 6:
            segments.append(audio[start * 1000:i * 1000])
            start = i
    segments.append(audio[start * 1000:])

    # Normalize the volume in each segment
    for i, segment in enumerate(segments):
        volume_ratio = volume_ratio - 25
        normalized_segment = segment.apply_gain(volume_ratio)
        print('ratio:', volume_ratio, ' | orig:', segment.dBFS, 'normalized: ', normalized_segment.dBFS)
        segments[i] = normalized_segment

    # Concatenate the normalized segments and export the result
    output = segments[0]
    for segment in segments[1:]:
        output = output + segment
    output.export(output_file, format="wav")


def increase_vol(input_audio, output_audio):
    # Load the input file
    input_file = AudioSegment.from_wav(input_audio)

    # Increase the volume by 13 dB
    output_file = input_file + 13

    # Save the output file
    output_file.export(output_audio, format='wav')


def main():
    inputFile = 'D:\Projects\kion_safty\demos\SCREAMER.mp4'
    outputFile = 'D:\Projects\kion_safty\demos\CREAMER.wav'
    normFile = 'D:\Projects\kion_safty\demos\CREAMER_normalized.wav'

    if not exists(outputFile):
        extract_wav(inputFile, outputFile)

    # find loudness in file
    data, rate = sf.read(outputFile)  # load audio (with shape (samples, channels))

    meter = pyln.Meter(rate)  # create BS.1770 meter
    # create a meter initialized without filters

    # load your filters into the meter
    loudness = meter.integrated_loudness(data)  # measure loudness
    print('loudness', loudness)

    normalize_audio(inputFile, outputFile)
    increase_vol(outputFile, normFile)

    # loudness normalize audio to -12 dB LUFS
    # loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -12.0)
    # sf.write(normFile, loudness_normalized_audio, rate)
    print('Done.')

    # peak_normalize_audio(outputFile, normFile)


if __name__ == '__main__':
    main()
