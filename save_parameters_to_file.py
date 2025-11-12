#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 12:07:46 2025

@author: emilio
"""

from datetime import datetime

def save_parameters_to_file(useSecondAD2, GR_On, shim_On, numAvgs, frequency, amplitude, TE, Tp, Npulse, TR1,
                            sampFreq, Tacq, IF_freq, amplitude_LO, Npts, FOV, ramp_time, calibration_factor):
    
    # Date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Lab 9 Proj/mri_echo_{timestamp}_params.npz"

    # Write the parameters to the file
    with open(filename, 'w') as file:
        file.write(f"useSecondAD2 = {useSecondAD2}\n")
        file.write(f"GR_On = {GR_On}\n")
        file.write(f"shim_On = {shim_On}\n")
        file.write(f"numAvgs = {numAvgs}\n")
        file.write(f"frequency = {frequency}\n")
        file.write(f"amplitude = {amplitude}\n")
        file.write(f"TE = {TE}\n")
        file.write(f"Tp = {Tp}\n")
        file.write(f"Npulse = {Npulse}\n")
        file.write(f"TR1 = {TR1}\n")
        file.write(f"sampFreq = {sampFreq}\n")
        file.write(f"Tacq = {Tacq}\n")
        file.write(f"IF_freq = {IF_freq}\n")
        file.write(f"amplitude_LO = {amplitude_LO}\n")
        file.write(f"Npts = {Npts}\n")
        file.write(f"FOV = {FOV}\n")
        file.write(f"ramp_time = {ramp_time}\n")
        file.write(f"calibration_factor = {calibration_factor}\n")

    print(f"Parameters saved to {filename}")

