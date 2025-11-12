#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 21:54:59 2025

@author: emilio
"""


def check_shim_voltage(shim_V):
    # Check voltage limits
    if abs(shim_V) > 0.1:
        raise ValueError(f"Shim voltage limit exceeded! "
                         f"shim_V={shim_V:.3f} V. "
                         "Voltages must stay within ±0.1 V.")
    else:
        print(f"Shim voltages OK: shim_V={shim_V:.3f} V")
