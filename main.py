'''
Starting code for the Waveforms SDK.    MR Engineering Class
Lab 6
Notes: 
       Pin 0 is set to be an external scope trigger
       Pin 2 is the T/R switch conotrol line
       Pin 3 is the attenuator control line
       Pin 4 is connected to external trigger input 1, to trigger the AD2 digitizer
       
'''

from dwfconstants import *

import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from scipy.fft import fft, fftfreq, fftshift
from scipy import signal
from numpy.fft import fft, fftshift

##   Hide the SDK functions by left click next to the if statement
import_mre_functions = 1
if import_mre_functions == 1:
    
  
    def set_dio(ChNum,totalCycles,low,high):
        #   The DIO can be set by the number of cycles low, then high, and then low again. 
        #   Input values are in seconds, then converted to cycles assuming 1 clock cycle/ 10 microseconds.  
        DIOLow1 = int(low * 10**5)
        DIOHigh = int(high * 10**5)
        DIOLow2 = int(totalCycles - DIOHigh - DIOLow1)
        dwf.FDwfDigitalOutEnableSet(hdwf, c_int(ChNum), c_int(1))
        dwf.FDwfDigitalOutDividerSet(hdwf, c_int(ChNum), c_int(int(hzSys.value / 100000)))
        dwf.FDwfDigitalOutCounterSet(hdwf, c_int(ChNum), c_int(DIOLow2), c_int(DIOHigh))
        dwf.FDwfDigitalOutCounterInitSet(hdwf, c_int(ChNum), c_int(0), c_int(DIOLow1))
        dwf.FDwfDigitalOutIdleSet(hdwf, c_int(ChNum), DwfDigitalOutIdleLow)
        return 
    
    def set_scope(sampFreq,numSamp,acqTime,Delay):
        dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeSingle)  # set to a single acquisition
        dwf.FDwfAnalogInFrequencySet(hdwf, c_double(sampFreq))  # sets up the frequency
        dwf.FDwfAnalogInBufferSizeSet(hdwf, c_int(numSamp))  # sets the buffer
        dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_bool(True))  # enables channel 0
        dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(1), c_bool(False))  # disable  channel 1
        dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(-1), c_double(5))  # sets the range
        dwf.FDwfAnalogInChannelFilterSet(hdwf, c_int(-1), filterDecimate)  
        dwf.FDwfAnalogInTriggerSourceSet(hdwf, trigsrcExternal1)  # sets the trigger source
        dwf.FDwfAnalogInTriggerConditionSet(hdwf, DwfTriggerSlopeRise)
        dwf.FDwfAnalogInTriggerPositionSet(hdwf, c_double(acqTime / 2 + Delay) ) # sets the trigger position
        y = 0
        return y
    
    def set_wavegen(ChNum,freq,amplitude,pulseL,pd,Nreps):
        dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(ChNum), AnalogOutNodeCarrier, c_bool(True))
        dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(ChNum), AnalogOutNodeCarrier, funcSine)  # Function
        dwf.FDwfAnalogOutNodeFrequencySet(hdwf, c_int(ChNum), AnalogOutNodeCarrier, c_double(freq))  # frequency
        dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, c_int(ChNum), AnalogOutNodeCarrier, c_double(amplitude))  # Amplitude
        dwf.FDwfAnalogOutRunSet(hdwf, c_int(ChNum), c_double(pulseL))  # run time
        dwf.FDwfAnalogOutWaitSet(hdwf, c_int(ChNum), c_double(pd))  # wait time
        dwf.FDwfAnalogOutRepeatSet(hdwf, c_int(ChNum), c_int(Nreps))  # repetitions
#        dwf.FDwfAnalogOutTriggerSourceSet(hdwf, c_int(ChNum), trigsrcExternal1)  # sets the trigger source        
        dwf.FDwfAnalogOutTriggerSourceSet(hdwf, c_int(ChNum), trigsrcDigitalOut)  # sets the trigger source
        y = 0
        return y   

    def set_pos_powersupply(Voltage):
        dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(0), c_double(True))  # enable positive supply
        dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(1), c_double(Voltage))  # set voltage to 5 V
        dwf.FDwfAnalogIOEnableSet(hdwf, c_int(True))  # master enable
        y = 0
        return y
 
    def arm_dio(totalTime):
        # Finishing setting up the DIO pins.
        dwf.FDwfDigitalOutRunSet(hdwf, c_double(totalTime))
        dwf.FDwfDigitalOutWaitSet(hdwf, c_double(0))
        dwf.FDwfDigitalOutRepeatSet(hdwf, c_int(1))
        y = 0
        return y
    
    def trigger_and_read_ch0(rgdSamples,numSamp):
        dwf.FDwfDigitalOutConfigure(hdwf, c_int(1))
        while True:
            dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
            if sts.value == DwfStateDone.value:
                break
        dwf.FDwfAnalogInStatusData(hdwf, 0, rgdSamples, numSamp)  # get channel 1 data
        
        y = 0
        return y
    
    def trigger_and_read_ch1(rgdSamples,numSamp):
        dwf.FDwfDigitalOutConfigure(hdwf, c_int(1))
        while True:
            dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
            if sts.value == DwfStateDone.value:
                break
        dwf.FDwfAnalogInStatusData(hdwf, 1, rgdSamples, numSamp)  # get channel 2 data
        
        y = 0
        return y
    
    def arm_analog():
        dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(1))
        dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_bool(True))
        dwf.FDwfAnalogOutConfigure(hdwf, c_int(1), c_bool(True))
        y = 0
        return y
    
    def set_ad2_device(idevice):
        dwf.FDwfEnumDeviceName(c_int(idevice), devicename)
        dwf.FDwfEnumSN(c_int(idevice), serialnum)
        hdwf.value = rghdwf[idevice]
        y = 0
        return y
    
    def reset_and_close():
        dwf.FDwfDigitalIOReset()
        dwf.FDwfDeviceCloseAll()
        y = 0
        return y


###################################################################
# CONTROLABLE PARAMETERS
###################################################################


numAvgs = 10

# RF pulse paramters
frequency = 3312666
amplitude = 5
TE = .02
Tp = .0005

Npulse = 2   
TR1 = 0.25

# Data acquisition parameters
sampFreq = 1000000
Tacq = .0064

# LO wave parameters
IF_freq = 100000 # 100kHz
amplitude_LO = 0.5

# Gradient parameters
Npts = 60
FOV = 2    # cm
ramp_time1 = 0.0005
calibration_factor = 1.375  # G/cm/V_ad2


# Print out important data
print(f"RF pulse frequency is {frequency} MHz.")

###################################################################
# COMPUTED PARAMETERS
###################################################################

# RF pulse paramters
predelay = TE / 2 - Tp

# Data acquisition parameters
numSamp = int(Tacq * sampFreq)
Trig_AD2 = (TE + Tp/2 + predelay) - 0.5 * Tacq  #  trigger the AD2 digitizer at TE - Tacq/2
SeqTime = .04    # duration that will encompass a single pulse sequence
DIO_rate = 10**5  # effective clock rate of the digital i/O
totalCycles = SeqTime*DIO_rate

# LO wave parameters
frequency_LO = frequency - IF_freq
LO_buffer_time = 0.001
Tp_LO = Tacq + LO_buffer_time
predelay_LO = Trig_AD2 - LO_buffer_time/2.
Npulse_LO = 1

# Gradient parameters
ramp_time1 = 0.0005                  # ramp up/down time (s)
total_time = TE + Tacq/2 + ramp_time1       # total waveform duration in seconds
freq_resolution = 1/Tacq                        # Hz
img_res = (FOV * 10.) / Npts                    # mm
gradient_strength_Hzmm = freq_resolution / img_res      # Hz/mm
gradient_strength_Gcm = gradient_strength_Hzmm / 425.7  # G/cm
V_grad = gradient_strength_Gcm / calibration_factor 
V_grad_ad2 = V_grad / 11                # correct for amp gain of 11

print(f'AD2 voltage is {V_grad_ad2}')
print(f'Gradient voltage is {V_grad}')
print(f'Current into the gradient coil is {V_grad/4} amps')
print(f'Gradient strength is {gradient_strength_Gcm} G/cm')
print(f'Image resolution is {img_res}')
print(f'Dephase pulse length is half Tacq = {Tacq/2}.')
print(f'Dephase pulse has voltage amplitude same as FE pulse of {V_grad}')

###################################################################
# Bandpass filter parameters
f_center = 100000   # Mixed-down center frequency (IF)
bw = 10000          # Total bandwidth (±10 kHz)

f_low = f_center - bw // 2   # 95,000 Hz
f_high = f_center + bw // 2  # 105,000 Hz

butterworth_N = 1  # Filter order: higher = sharper rolloff
butterworth_Wn = [f_low, f_high]
    

###################################################################
        # Begin transmit/receive
##################################################################

signals_total = np.zeros(numSamp)

for i in range(numAvgs):  
    
    
    ###################################################################
           # Opens the AD2s
    ##################################################################
    
    ##   Hide the open_ad2 code by by left click next to the if statement
    open_ad2 = 1
    prt_info = 1
    if open_ad2 == 1:
        dwf = cdll.LoadLibrary('/Library/Frameworks/dwf.framework/dwf')
        # check library loading errors, like: Adept Runtime not found
        szerr = create_string_buffer(512)
        dwf.FDwfGetLastErrorMsg(szerr)
        # declare ctype variables
        IsInUse = c_bool()
        hdwf = c_int()
        rghdwf = []
        cchannel = c_int()
        cdevices = c_int()
        voltage = c_double();
        sts = c_byte()
        hzAcq = c_double(sampFreq)  # changes sample frequency into c_double
        rgdSamples = (c_double * numSamp)()  # list for C1 on scope
        # declare string variables
        devicename = create_string_buffer(64)
        serialnum = create_string_buffer(16)
      
        # enumerate connected devices
        dwf.FDwfEnum(c_int(0), byref(cdevices))
    #            print ("Number of Devices: "+str(cdevices.value))
        
        # open and configure devices
        for idevice in range(0, cdevices.value):
            dwf.FDwfEnumDeviceName(c_int(idevice), devicename)
            dwf.FDwfEnumSN(c_int(idevice), serialnum)
            if (prt_info == 1):
              print ("------------------------------")
        #              print (' idevice = ',idevice)
              print ("Device "+str(idevice+1)+" : ")
              print ('Serial Number = ',serialnum.value)
            dwf.FDwfDeviceOpen(c_int(idevice), byref(hdwf))
            if hdwf.value == 0:
                szerr = create_string_buffer(512)
                dwf.FDwfGetLastErrorMsg(szerr)
                print (szerr.value)
                dwf.FDwfDeviceCloseAll()
                sys.exit(0)
                
            rghdwf.append(hdwf.value)           
        # looks up buffer size
            cBufMax = c_int()
            dwf.FDwfAnalogInBufferSizeInfo(hdwf, 0, byref(cBufMax))
            
            dwf.FDwfEnumDeviceName(c_int(idevice), devicename)
            dwf.FDwfEnumSN(c_int(idevice), serialnum)
            hdwf.value = rghdwf[idevice]
        # configure and start clock
        hzSys = c_double()
        dwf.FDwfDigitalOutInternalClockInfo(hdwf, byref(hzSys))
    #  Finished setting up multiple AD2s
    #############################################################
    
    y1 = set_ad2_device(0)
    
    # Setup External Scope trigger
    Trig_low = .0001
    Trig_high = .001
    y = set_dio(0,totalCycles,Trig_low,Trig_high)
    
    # Setup AD2 Scope trigger
    Trig_low = Trig_AD2
    Trig_high = Tacq
    y = set_dio(4,totalCycles,Trig_low,Trig_high)
    
    
    
    ################################################################
    # Gradient waveform generator
    ################################################################
    
    # === PARAMETERS ===
    cSamples = 8196
    deltaT = total_time / cSamples
    channel = c_int(0)
    hzFreq2 = 1/total_time
    
    # Maximum voltage
    max_voltage = 1.0
    
    # === TRAPEZOID SPECIFICATIONS ===
    # Trapezoid 1 (starts at beginning)
    #ramp_time1 = 0.001        
    hold_time1 = Tacq/2        # flat top time (s)
    
    # Trapezoid 2 (can be placed anywhere by midpoint)
    ramp_time2 = ramp_time1
    hold_time2 = Tacq
    midpoint_time2 = TE - Tp/2
    
    # === CREATE TIME ARRAY ===
    time_array = np.arange(0, total_time, deltaT)
    
    # Initialize waveform with zeros
    rgdSamples2_np = np.zeros(cSamples)
    
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
    
    # === CREATE TRAPEZOIDS ===
    trapezoid1 = create_trapezoid(ramp_time1, hold_time1, max_voltage, deltaT)
    trapezoid2 = create_trapezoid(ramp_time2, hold_time2, max_voltage, deltaT)
    
    # Insert trapezoids
    rgdSamples2_np = insert_trapezoid(rgdSamples2_np, trapezoid1, start_index=0)
    
    # Compute start index for trapezoid2 based on midpoint
    center_index2 = int(midpoint_time2 / deltaT)
    start_index2 = center_index2 - len(trapezoid2)//2
    rgdSamples2_np = insert_trapezoid(rgdSamples2_np, trapezoid2, start_index2)
    
    # === CONVERT TO CTYPE ARRAY FOR HARDWARE ===
    rgdSamples2 = (c_double * cSamples)()
    for i in range(cSamples):
        rgdSamples2[i] = rgdSamples2_np[i]
    
    # === PLOT ===
    # plt.figure(figsize=(10,4))
    # plt.plot(time_array, rgdSamples2_np)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Voltage')
    # plt.grid(True)
    # plt.show()
    
    #######################################################################
    y1 = set_ad2_device(1)
    #######################################################################
    
    

    print("Generating custom waveform...")
    dwf.FDwfAnalogOutNodeEnableSet(hdwf, channel, AnalogOutNodeCarrier, c_int(1))
    dwf.FDwfAnalogOutNodeFunctionSet(hdwf, channel, AnalogOutNodeCarrier, funcCustom) 
    dwf.FDwfAnalogOutNodeDataSet(hdwf, channel, AnalogOutNodeCarrier, rgdSamples2, c_int(cSamples))
    dwf.FDwfAnalogOutNodeFrequencySet(hdwf, channel, AnalogOutNodeCarrier, c_double(hzFreq2)) 
    dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, channel, AnalogOutNodeCarrier, c_double(V_grad_ad2)) 

    dwf.FDwfAnalogOutRunSet(hdwf, channel, c_double(1.0/hzFreq2)) # run for 2 periods
    dwf.FDwfAnalogOutWaitSet(hdwf, channel, c_double(predelay + Tp)) # wait one pulse time
    dwf.FDwfAnalogOutRepeatSet(hdwf, channel, c_int(1)) # repeat 3 times
    dwf.FDwfAnalogOutTriggerSourceSet(hdwf, channel, trigsrcExternal1)  # sets the trigger source
    
    

    
    ########################################################################
    # Shim Control
    ########################################################################
    
    
    shim1_ChNum = 1
    shim1_V = 0.
        
        # Check voltage limits
    if abs(shim1_V) > 0.1:
        raise ValueError(f"Shim voltage limit exceeded! "
                         f"shim1_V={shim1_V:.3f} V. "
                         "Voltages must stay within ±0.1 V.")
    else:
        print(f"Shim voltages OK: shim1_V={shim1_V:.3f} V")
    
    
    print("Generating shim voltage...")
    dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(shim1_ChNum), c_int(1)) 
    dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(shim1_ChNum), funcDC) # Function 
    dwf.FDwfAnalogOutOffsetSet(hdwf, c_int(shim1_ChNum), c_double(shim1_V))
    
    
    y2 = set_pos_powersupply(5)
    y2 = arm_analog()
    #######################################################################
    y1 = set_ad2_device(0)
    #######################################################################
    
    
    ########################################################################
    # Control lines
    ########################################################################
    
    
    # ======= T/R control line =======
    y = set_dio(2,totalCycles,predelay - 0.5 * Tp, predelay + 3 * Tp)
    
    
    # ======= Attenuator control line =======
    y = set_dio(3,totalCycles,predelay - 0.5 * Tp, 2 * Tp)
    
    ########################################################################

    ########################################################################
    # Wavegens
    ########################################################################
    
    
    # turn on positive voltage power supply
    y2 = set_pos_powersupply(5)
    
    # Set up the RF pulse generator
    y1 = set_wavegen(0,frequency,amplitude,Tp,predelay,Npulse) # single pair of RF pulse
    
    
    # Set up the LO waveform
    y1 = set_wavegen(1,frequency_LO, amplitude_LO, Tp_LO, predelay_LO, Npulse_LO) 
    
    # set up acquisition (scope) (Lab 2)
    delay = 0.0
    y1 = set_scope(sampFreq,numSamp,Tacq,delay) 
    
    # Arm the analog and digital sections 
    y1 = arm_dio(SeqTime)
    y1 = arm_analog()      
    time.sleep(1)
    
    #  trigger and collect data
    print('going to trigger')
    y1 = trigger_and_read_ch0(rgdSamples,numSamp)      
    print('back from trigger')  
    
    ## define time array for plot
    Tacqms = Tacq * 1000.
    time2 = np.linspace(0, Tacqms, numSamp) # linspace(start_time, total time, number of points)
    
    ## Butterworth bandpass filter
    sos = signal.butter(butterworth_N, butterworth_Wn, 'bandpass', fs=sampFreq, output='sos')
    filtered = signal.sosfilt(sos, rgdSamples)
        
    # ========== Plot acquired data ================
    plt.plot(time2, rgdSamples)
    plt.xlabel('Time msec')
    plt.ylabel('Measured Voltage (V)')
    plt.title('Signal')
    fig1 = plt.show()
    
    # plt.plot(time2, filtered)
    # plt.xlabel('Time msec')
    # plt.ylabel('Measured Voltage (V)')
    # plt.title('Filtered Signal')
    # fig2 = plt.show()
    
    signals_total = [float(rgdSamples[j]) + signals_total[j] for j in range(len(signals_total))]
        
    y1 = reset_and_close()
    time.sleep(TR1)
    
signals_avg = [val / numAvgs for val in signals_total]

plt.plot(time2, signals_avg)
plt.xlabel('Time (ms)')
plt.ylabel('Measured Voltage')
plt.title(f'Averaged Signal with {numAvgs} Averages')
#plt.title(f'Summed Signal with {numAvgs} Sums')
fig3 = plt.show()
    
# ========== Plot FFT ================
npts = numSamp
deltat = 1e-6  # 1 microsecond = 1 MSps sampling rate

# FFT calculation
y_transform = fftshift(fft(fftshift(signals_avg)))

# Frequency axis
deltaf = 1 / (npts * deltat)
fmax = deltaf * npts / 2
fmin = -fmax + deltaf
freq = np.linspace(fmin, fmax, npts)

# Center frequency and bandwidth
center_freq = 1e5  # 100 kHz
bandwidth = 4e4    # 20 kHz
lower_bound = center_freq - bandwidth / 2
upper_bound = center_freq + bandwidth / 2

# Find indices within desired frequency range
mask = (freq >= lower_bound) & (freq <= upper_bound)

# Plot only the desired band
plt.plot(freq[mask], abs(y_transform[mask]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('FFT Magnitude')
plt.title(f'FFT around {center_freq/1e3:.0f} kHz (±{bandwidth/2/1e3:.0f} kHz) - RF Freq {frequency}')
plt.grid(True)
plt.show()
       
# Write data to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"mri_echo_{timestamp}.npz"

# ========= Save data to npz file ================
np.savez(
    filename,
    averaged_signal=signals_avg,
    avgs = numAvgs,
    rf_freq=frequency,
    timestamp=timestamp
)

# time.sleep(2)
