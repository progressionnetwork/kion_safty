#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from os.path import join as path_join
import numpy as np
from scipy.io import wavfile
import scipy.signal as si
import matplotlib.pyplot as plt


def factor2decibel(factor):
    """Compute level from factor."""
    assert np.all(factor >= 0.0)
    return 20.0 * np.log10(np.abs(factor))


def true_peak(data, oversample=4):
    """Return the true-peak value of all channels. The absolute value is
    returned.
    """
    sample_count, _channels = data.shape
    upsampled_data = si.resample(data, sample_count * oversample, axis=0)
    result = np.amax(np.abs(upsampled_data), axis=None)
    return result


def wave_to_float64(filepath):
    """Read WAVE and fix data type and shape. Values are in [-1, 1]."""
    data = None
    try:
        samplerate, data = wavfile.read(filepath)

        # Fix range
        if not str(data.dtype).startswith('float'):
            info = np.iinfo(data.dtype)
            data = np.array(data, dtype=np.float32) / float(info.max)

        # Fix mono shape
        if len(data.shape) == 1:
            data = np.reshape(data, (-1, 1))

    except ValueError as err:
        # e.g. problem with 24 bits
        print(f"ERROR: {filepath} {err}")
        sys.exit(1)

    return samplerate, data


def visualize_levels(dbtps, rmss, dc_offsets):
    """Graph histogram of True-peak and RMS."""
    for lst, lbl in [(dbtps, "True-peak"), (rmss, "RMS"), (dc_offsets, "DC offset")]:
        ary = np.array(lst)
        hist, bin_edges = np.histogram(ary, bins='auto', density=False)
        bin_width = np.diff(bin_edges)
        mid = bin_edges[:-1] + (bin_width / 2.0)
        plt.step(mid, hist, where='mid', label=lbl)

    plt.title("Dirt-Samples: True-peak, RMS and DC-offset of {} samples".format(len(dbtps)))
    plt.xlabel("Level [dBFS]")
    plt.ylabel("Count")

    plt.legend()
    plt.show()


def analyse(filepath):
    """Analyse True-peak and RMS of the given file."""
    dbtp = None
    dbrms = None
    dbdc_offset = None

    if filepath.lower().endswith('.wav'):
        result = wave_to_float64(filepath)
        if not result is None:
            _samplerate, data = result

            tp = true_peak(data, oversample=4)
            dbtp = factor2decibel(tp)

            rms = np.sqrt(np.mean(np.square(data), axis=None))
            dbrms = factor2decibel(rms)

            dc_offset = np.abs(np.mean(data, axis=None))
            dbdc_offset = np.max([-147.0, factor2decibel(dc_offset)])
        else:
            pass
    else:
        pass

    return dbtp, dbrms, dbdc_offset


def file_tree_map(fun, start):
    """Traverse a file tree and call `fun(filepath)`. `start` can be a
    relative path, e.g. the working directory `.`. `filepath` is an
    absolute path.
    """
    dbtps = []
    rmss = []
    dc_offsets = []

    for root, _dirs, files in os.walk(start):
        for filename in files:
            filepath = path_join(root, filename)
            dbtp, rms, dc_offset = fun(filepath)
            if not dbtp is None and not rms is None:
                dbtps.append(dbtp)
                rmss.append(rms)
                dc_offsets.append(dc_offset)

    visualize_levels(dbtps, rmss, dc_offsets)


def file_list_map(fun, filepaths):
    """Traverse a list of filepaths, apply `fun` to filepath."""
    for filepath in filepaths:
        fun(filepath)


def main():
    filepaths = [
        './wobble/000_0.wav',
    ]
    file_list_map(analyse, filepaths)


if __name__ == '__main__':
    main()
