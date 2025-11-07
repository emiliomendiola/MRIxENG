#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 21:01:40 2025

@author: emilio
"""

import numpy as np
from ctypes import *
from dwfconstants import *
import matplotlib.pyplot as plt

# --- Create the waveform ---

def createGradientWaveform(gradient_amplitude, ramp_time, hold_time, hzFreq2=None, cSamples=4096):
    """
    Create a ramp-up, hold, and ramp-down waveform filling exactly cSamples samples,
    with each time segment corresponding to real physical durations.
    
    Parameters:
        gradient_amplitude (float): Maximum amplitude (0–1 range typically)
        ramp_time (float): Ramp-up/down duration in seconds
        hold_time (float): Hold duration in seconds
        hzFreq2 (float | None): Sampling rate in Hz. If None, it will be computed
                                to match total waveform duration.
        cSamples (int): Total number of samples in the waveform
        
    Returns:
        waveform (np.ndarray): The generated waveform (length = cSamples)
        hzFreq2 (float): The actual sampling frequency (Hz)
    """

    # --- 1. Total waveform duration (in seconds)
    total_time = 2 * ramp_time + hold_time

    # --- 2. Determine the sample rate (Hz)
    if hzFreq2 is None:
        hzFreq2 = cSamples / total_time  # ensures waveform fits exactly in given duration

    # --- 3. Compute number of samples for each segment
    ramp_samples = int(round(ramp_time * hzFreq2))
    hold_samples = int(round(hold_time * hzFreq2))

    # adjust to fill entire array
    total_assigned = 2 * ramp_samples + hold_samples
    if total_assigned != cSamples:
        hold_samples += cSamples - total_assigned  # small fix to ensure exact length

    # --- 4. Create waveform parts
    ramp_up = np.linspace(0, gradient_amplitude, ramp_samples, endpoint=False)
    hold = np.full(hold_samples, gradient_amplitude)
    ramp_down = np.linspace(gradient_amplitude, 0, ramp_samples, endpoint=True)

    waveform = np.concatenate([ramp_up, hold, ramp_down])

    # --- 5. Safety adjustment for rounding errors
    if len(waveform) > cSamples:
        waveform = waveform[:cSamples]
    elif len(waveform) < cSamples:
        waveform = np.pad(waveform, (0, cSamples - len(waveform)), 'constant')

    return waveform, hzFreq2

# --- Configure and output waveform ---
def configure_custom_gradient_waveform(
    dwf,
    hdwf,
    createGradientWaveform,
    Tacq,
    TE,
    Tp,
    predelay,
    V_grad_ad2,
    grad_time_delay,
    gradient_amplitude=1.0,
    hold_time=0.001,
    ramp_time=0.001,
    cSamples=4096,
    channel_index=1,
):
    """Configure and output a custom gradient waveform using a DWF device."""
    
    # --- Derived times ---
    total_time = 2 * ramp_time + hold_time
    channel = c_int(channel_index)
    
    # --- Generate waveform ---
    waveform, hzFreq2 = createGradientWaveform(gradient_amplitude, ramp_time, hold_time, None, cSamples)
    
    print(f"Generated waveform: total time = {total_time:.6f} s, "
          f"sample rate = {hzFreq2:.1f} Hz, total samples = {cSamples}")
    
    # --- Convert to C array ---
    rgdSamples2 = (c_double * cSamples)()
    for i in range(cSamples):
        rgdSamples2[i] = waveform[i]
    
    plt.plot(waveform)
    plt.show()
    
    print("Configuring custom gradient waveform...")
    waveform_freq = 1.0 / total_time
    
    print(f"Waveform frequency set to {waveform_freq:.3f} Hz "
          f"for {total_time * 1e3:.3f} ms total duration")
    
    # --- Configure AWG output ---
    dwf.FDwfAnalogOutNodeEnableSet(hdwf, channel, AnalogOutNodeCarrier, c_int(1))
    dwf.FDwfAnalogOutNodeFunctionSet(hdwf, channel, AnalogOutNodeCarrier, funcCustom)
    dwf.FDwfAnalogOutNodeDataSet(hdwf, channel, AnalogOutNodeCarrier, rgdSamples2, c_int(cSamples))
    dwf.FDwfAnalogOutNodeFrequencySet(hdwf, channel, AnalogOutNodeCarrier, c_double(waveform_freq))
    dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, channel, AnalogOutNodeCarrier, c_double(V_grad_ad2))
    
    dwf.FDwfAnalogOutRunSet(hdwf, channel, c_double(total_time))
    dwf.FDwfAnalogOutWaitSet(hdwf, channel, c_double(grad_time_delay))
    dwf.FDwfAnalogOutRepeatSet(hdwf, channel, c_int(1))
    dwf.FDwfAnalogOutTriggerSourceSet(hdwf, channel, trigsrcDigitalOut)
    
    print("Custom waveform ready for output.")
