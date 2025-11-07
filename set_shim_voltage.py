#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 21:54:59 2025

@author: emilio
"""

from dwfconstants import *

dwf = cdll.LoadLibrary('/Library/Frameworks/dwf.framework/dwf')

def set_shim(ChNum,offset):
    dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(ChNum), c_int(1))
    dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(ChNum), funcDC) # Function 
    dwf.FDwfAnalogOutOffsetSet(hdwf, c_int(ChNum), c_double(offset))
    y=0
    
    return y
