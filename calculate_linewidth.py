#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 22:24:16 2025

@author: emilio
"""

import numpy as np

def calculate_linewidth(frequencies, spectrum):
    """
    Calculate the linewidth (FWHM) of a peak in a spectrum.

    Parameters
    ----------
    frequencies : np.ndarray
        1D array of frequencies corresponding to the spectrum (Hz).
    spectrum : np.ndarray
        1D array of magnitudes (or power) of the spectrum.

    Returns
    -------
    linewidth : float
        Linewidth (full width at half maximum) in Hz.
    peak_freq : float
        Frequency of the peak.
    """
    # Ensure arrays are numpy arrays
    frequencies = np.array(frequencies)
    spectrum = np.array(spectrum)

    # Find peak
    peak_idx = np.argmax(spectrum)
    peak_mag = spectrum[peak_idx]
    half_max = peak_mag / 2
    peak_freq = frequencies[peak_idx]

    # Find the points where spectrum crosses half max on each side
    # Left side
    left_idx = peak_idx
    while left_idx > 0 and spectrum[left_idx] > half_max:
        left_idx -= 1

    # Interpolate for better accuracy
    if left_idx > 0:
        f_left = np.interp(half_max, [spectrum[left_idx], spectrum[left_idx + 1]],
                           [frequencies[left_idx], frequencies[left_idx + 1]])
    else:
        f_left = frequencies[0]

    # Right side
    right_idx = peak_idx
    while right_idx < len(spectrum)-1 and spectrum[right_idx] > half_max:
        right_idx += 1

    # Interpolate for better accuracy
    if right_idx < len(spectrum)-1:
        f_right = np.interp(half_max, [spectrum[right_idx - 1], spectrum[right_idx]],
                            [frequencies[right_idx - 1], frequencies[right_idx]])
    else:
        f_right = frequencies[-1]

    # Linewidth
    linewidth = f_right - f_left
    return linewidth, peak_freq