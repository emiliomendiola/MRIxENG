#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 21:26:33 2025

@author: emilio
"""

import numpy as np

def snr_from_fft(fft_data, peak_width=5):
    """
    Calculate the Signal-to-Noise Ratio (SNR) from an FFT spectrum.
    
    Parameters
    ----------
    fft_data : np.ndarray
        The FFT of the signal (complex or magnitude). Can be one-sided or two-sided.
    peak_width : int, optional
        Number of bins around the peak to consider as part of the signal.
        
    Returns
    -------
    snr_db : float
        Signal-to-Noise Ratio in decibels (dB).
    """
    # Convert to magnitude if complex
    magnitude = np.abs(fft_data)
    
    # Find the index of the maximum peak
    peak_idx = np.argmax(magnitude)
    
    # Define the region around the peak considered as "signal"
    lower = max(0, peak_idx - peak_width)
    upper = min(len(magnitude), peak_idx + peak_width + 1)
    
    # Compute signal power and noise power
    signal_power = np.sum(magnitude[lower:upper] ** 2)
    noise_indices = np.ones(len(magnitude), dtype=bool)
    noise_indices[lower:upper] = False
    noise_power = np.std(magnitude[noise_indices])
    
    # Avoid division by zero
    if noise_power == 0:
        return np.inf
    
    # Compute SNR in dB
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    return snr_db