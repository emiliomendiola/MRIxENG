#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 21:01:40 2025

@author: emilio
"""

import numpy as np

def create_trapezoid(ramp_time, hold_time, max_voltage, deltaT):
    """Return one trapezoid waveform array."""
    ramp_samples = int(ramp_time / deltaT)
    hold_samples = int(hold_time / deltaT)
    
    ramp_up = np.linspace(0, max_voltage, ramp_samples, endpoint=False)
    hold = np.ones(hold_samples) * max_voltage
    ramp_down = np.linspace(max_voltage, 0, ramp_samples, endpoint=False)
    
    return np.concatenate((ramp_up, hold, ramp_down))

def insert_trapezoid(waveform, trapezoid, start_index):
    """Insert trapezoid into waveform at a specified start index."""
    n = len(trapezoid)
    end_index = start_index + n
    
    # Make sure indices are within waveform bounds
    if end_index > len(waveform):
        trapezoid = trapezoid[:len(waveform)-start_index]
        end_index = len(waveform)
    
    waveform[start_index:end_index] += trapezoid
    return waveform

