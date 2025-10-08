#!/usr/bin/env python
# coding: utf-8
# PURPOSE: OFDM RealTime Signal Receiver
# Aug. 2025

import time
import uhd
import numpy as np
import argparse
from scipy.signal import correlate, correlation_lags, fftconvolve
from scipy.spatial.distance import hamming
from collections import deque
import scipy.io
import requests
from requests.auth import HTTPBasicAuth
import matplotlib.pyplot as plt

class RealTime_OFDM_Detector:
    def adjust_parameters_by_sample_rate(self):
        # Adjust chip_samples, FFT, CP based on sampling rate heuristics
        base_rate = 2e6  # default baseline for scaling
        scale = self.sample_rate / base_rate
        self.chip_samples = int(480 * scale)
        self.FFT = int(64 * scale)
        self.CP = int(16 * scale)
        self.OFDM_size = self.FFT + self.CP

        print(f"[Adjusted Parameters] chip_samples: {self.chip_samples}, FFT: {self.FFT}, CP: {self.CP}, OFDM_size: {self.OFDM_size}")

    def __init__(self, device_addr="addr=192.168.40.2", center_freq=3.385e9,
                 sample_rate=2e6, gain=30, cp=16, 
                 pseudonym_length=38, 
                 OFDM_size=80,  FFT=64, packet = 560):

        self.device_addr = device_addr
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.adjust_parameters_by_sample_rate()
        self.gain = gain
        self.detection_times = []
        
        
        self.OFDM_size = OFDM_size
        self.packet_length = packet
        self.FFT = FFT
        
        self.CP = cp
        
        self.ltf_preamble = self.generate_ltf()
        self.rx_samples = self.packet_length * 10 + len(self.ltf_preamble)
        self.buffer_length = int(2 * self.rx_samples)
        
        self.avg_signal_power = []
        self.avg_noise_power = []
        self.big_buffer = deque(maxlen=self.buffer_length)

        self.usrp = None
        self.rx_streamer = None
       
        # Subcarrier allocations
        self.pilotValue = 1.4142 + 1.4142j  # Pilot symbol
        self.allCarriers = np.arange(self.FFT)
        self.dataCarriers = np.array([-26,-25,-24,-23,-22,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,
                                      -9,-8,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,
                                      18,19,20,22,23,24,25,26])
        self.pilotCarriers = np.array([-21,-7,7,21])  # Pilot carriers

    def setup_usrp(self):
        self.usrp = uhd.usrp.MultiUSRP(self.device_addr)
        self.usrp.set_rx_rate(self.sample_rate)
        tune_request = uhd.types.TuneRequest(self.center_freq, 1.5e6)
        self.usrp.set_rx_freq(tune_request)
        self.usrp.set_rx_gain(self.gain)
        self.usrp.set_rx_antenna("TX/RX")
        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        self.rx_streamer = self.usrp.get_rx_stream(stream_args)
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(cmd)

    def stop_usrp(self):
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.rx_streamer.issue_stream_cmd(cmd)
        
    # Step 3: Your function to load and repeat it
    def generate_ltf(self):
        mat = scipy.io.loadmat('preamble.mat')
        ltf = mat['ltf'].flatten()  # complex array
        return np.tile(ltf, 2)      # repeat 2 times

    def cross_correlation_max(self, rx_signal, preamble_template):
        """
        Normalized complex cross-correlation to find preamble in received signal.
    
        Parameters:
            rx_signal (np.ndarray): Received complex IQ samples.
            preamble_template (np.ndarray): Known preamble sequence (complex IQ).
    
        Returns:
            preamble_start (int): Start index of detected preamble in rx_signal.
            data_start (int): Start index of data symbols (after preamble).
            ncc (np.ndarray): Normalized cross-correlation values.
        """
    
        # Matched filter: conjugate time-reverse of the template
        matched_filter = np.conj(preamble_template[::-1])
        corr = fftconvolve(rx_signal, matched_filter, mode="valid")
    
        # Energy of template (constant)
        template_energy = np.sqrt(np.sum(np.abs(preamble_template)**2)) + 1e-12
    
        # Sliding window RMS energy of received signal
        window_energy = np.sqrt(
            np.convolve(np.abs(rx_signal)**2,
                        np.ones(len(preamble_template), dtype=np.float32),
                        mode="valid")) + 1e-12
    
        # Normalized cross-correlation
        ncc = np.abs(corr) / (template_energy * window_energy)
    
        # Best match = index of maximum NCC
        preamble_start = int(np.argmax(ncc))
    
        # Data start = just after the preamble
        data_start = preamble_start + len(preamble_template)
    
        return preamble_start, ncc


    def average_power(self, signal):
        return np.abs(signal) ** 2


    def binvector2str(self, bits):
        """
        Convert a flat iterable of 0/1 to a 7-bit ASCII string (MSB first).
        Pure-Python, no NumPy required.
        """
        bits = list(bits)
        if len(bits) % 7 != 0:
            raise ValueError("Length of bit stream must be a multiple of 7.")
    
        out_chars = []
        for i in range(0, len(bits), 7):
            byte = 0
            # MSB first: positions 0..6 map to weights 64..1
            # converts the binary chunk into its integer value
            for b in bits[i:i+7]:
                byte = (byte << 1) | (1 if b else 0)
            out_chars.append(chr(byte))
        # Joins the list of characters into a single string
        text = ''.join(out_chars) 
        return text
        
    def mary2binary(self, data, M):
        length = len(data) # number of values in data
        log2M = round(np.log2(M)) # integer number of bits per data value
        format_string = '0' + str(log2M) + 'b'
        binarydata = np.zeros((1,length*log2M))
        count = 0
        for each in data:
            binval = format(int(each), format_string)
            for i in range(log2M):
                binarydata[0][count+i] = int(binval[i])
            count = count + log2M
        return binarydata
        
    def findClosestComplex(self, r_hat, outputVec):
        # outputVec is a 4-length vector for QPSK, would be M for M-QAM or M-PSK.
        # This checks, one symbol sample at a time,  which complex symbol value
        # is closest in the complex plane.
        data_out = [np.argmin(np.abs(r-outputVec)) for r in r_hat]
        return data_out

    def OFDM_RX(self, data):
        for i in range(len(data) // self.OFDM_size):
            data_cp = data[i*self.OFDM_size:(i+1)*self.OFDM_size]
            data_without_cp = data_cp[self.CP:]
            
            # Generate frequency domain signal
            OFDM_freq = np.fft.fft(data_without_cp, n=self.FFT)
            
            H_est = self.Channel_Estimation(OFDM_freq)  # estimate the channel
            OFDM_est = self.Equalization(OFDM_freq, H_est)  # sub-carrier equalization
            
            OFDM_data = OFDM_est[self.dataCarriers]  # extract the data signal
            
            if i == 0:
                OFDM_swap = OFDM_data
            else:
                OFDM_swap = np.concatenate((OFDM_swap, OFDM_data))
        return OFDM_swap

        
    def Channel_Estimation(self, signal):
        """
        Estimate the per-subcarrier channel H[k] for ONE OFDM symbol using pilots only.
        Strategy:
        """
        eps = 1e-12  # small number to prevent divide-by-zero / log(0)
        Yp = signal[self.pilotCarriers]
    
        # Xp: known transmitted pilot symbols.
        Xp = self.pilotValue if np.ndim(self.pilotValue) else np.full(len(self.pilotCarriers), self.pilotValue, dtype=complex)
    
        # Hp: channel estimate at pilot locations (divide received by known transmitted)
        Hp = Yp / (Xp + eps)
    
        # Split Hp into magnitude and phase for interpolation -
        mag_p = np.maximum(np.abs(Hp), eps)  # clamp to eps to avoid log(0)
        log_mag_p = np.log(mag_p)
    
        # Phase unwrap across pilot indices to avoid discontinuities at +/-pi boundaries.
        phase_p = np.unwrap(np.angle(Hp))
    
        # Interpolate (linear) from pilot bins to ALL ACTIVE subcarriers 
        log_mag_all = np.interp(self.allCarriers, self.pilotCarriers, log_mag_p)
        phase_all   = np.interp(self.allCarriers, self.pilotCarriers, phase_p)
    
        # Recombine to complex H on active carriers
        H_estimate = np.exp(log_mag_all) * np.exp(1j * phase_all)
    
        # Pack into a full FFT-sized vector, leaving guards/DC as zeros 
        H_full = np.zeros_like(signal, dtype=complex)
        H_full[self.allCarriers] = H_estimate
    
        return H_full

        
    def Equalization(self, OFDM_demod, Hest):
        return OFDM_demod / Hest
        
    def matched_filter_detection(self, rx_signal):
        outputVec = np.array([1+1j, -1+1j, 1-1j, -1-1j])
        time_signal = rx_signal[:self.packet_length]
        detected_signal = self.OFDM_RX(time_signal)
        mary_out  = self.findClosestComplex(detected_signal, outputVec)
        data_bits  = self.mary2binary(mary_out, 4)[0]
        return data_bits
    
    def refill_big_buffer(self):
        recv_buffer = np.zeros((1, self.rx_streamer.get_max_num_samps()-14), dtype=np.complex64)
        metadata = uhd.types.RXMetadata()
        while len(self.big_buffer) < self.buffer_length:
            num_rx = self.rx_streamer.recv(recv_buffer, metadata)
            if num_rx == 0 or metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                continue
            flat = recv_buffer.flatten()[:num_rx]
            self.big_buffer.extend(flat)

    def read_samples(self, num_needed):
        recv_buffer = np.zeros((1, self.rx_streamer.get_max_num_samps()-14), dtype=np.complex64)
        metadata = uhd.types.RXMetadata()
        while num_needed > 0:
            num_rx = self.rx_streamer.recv(recv_buffer, metadata)
            if num_rx == 0 or metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                continue
            flat = recv_buffer.flatten()[:num_rx]
            self.big_buffer.extend(flat)
            num_needed -= len(flat)

    def detect_ofdm(self):
        
        threshold_factor = 8.5
        corr_window_len = 80
    
        self.setup_usrp()
        self.refill_big_buffer()
        
        for i in range(10):
            samples = np.array(self.big_buffer, dtype=np.complex64)
            self.avg_noise_power.append(self.average_power(samples))
            self.refill_big_buffer()
            
        rx_noise =   np.mean(self.avg_noise_power)
        start_time = time.time()
        waiting_for_signal = False
        try:
            while True:
                samples = np.array(self.big_buffer, dtype=np.complex64)

                lag, corr = self.cross_correlation_max(samples[:-self.rx_samples], self.ltf_preamble)
                start = lag + len(self.ltf_preamble)

                corr_peak = np.max(np.abs(corr[lag:lag + corr_window_len]))
                corr_median = np.median(np.abs(corr))
                # ratio = corr_peak/corr_median
                
                if corr_peak < threshold_factor * corr_median:
                    self.read_samples(self.rx_samples)
                    if not waiting_for_signal:   # Only print once
                        print('Waiting for ofdm signal!')
                        waiting_for_signal = True
                    continue
                else:
                    waiting_for_signal = False  # Reset flag once signal is strong enough

                if start + self.rx_samples > len(samples):
                    self.read_samples(self.rx_samples)
                    print('🚫 Waiting for ofdm signal')
                    continue
    
                rx_signal = samples[start:start + self.packet_length]
                signal_power = self.average_power(rx_signal) - rx_noise
                self.avg_signal_power.append(signal_power)

                # Demodulate data symbols
                rx_bits = self.matched_filter_detection(rx_signal)
                rx_str = self.binvector2str(rx_bits)
                print("✅ Message Correctly Demodulated!")
                print('Received Message:',rx_str)

                # Write raw samples to file
                record = samples # [lag:start + self.packet_length]
                record = record.astype(np.complex64)
                # with open("rx_output.dat", "ab") as f:
                #     record.tofile(f)
                with open("rx_output.dat", "wb") as f:
                    record.tofile(f)
                    
                self.read_samples(self.rx_samples)
    
        except KeyboardInterrupt:
            print("Detection stopped.")
            
        finally:
            
            self.stop_usrp()
            
            # Ensure self.avg_signal_power has data before calculating SNR
            if len(self.avg_signal_power) > 0:
                snr = 10 * np.log10(0.5*np.mean(self.avg_signal_power) / (rx_noise +1e-12))
                print(f"SNR Estimate (48th subcarrier): {snr:.2f} dB")
            else:
                snr = 0
                print("No signal power data available. SNR cannot be calculated.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Watermark RX RealTime")
    parser.add_argument("-f", "--freq", type=float, default=3385e6, help="Center frequency in Hz (default: 3385e6)")
    parser.add_argument("-g", "--gain", type=float, default=30, help="Transmit gain in dB (default: 30)")
    parser.add_argument("-r", "--rate", type=float, default=2e6, help="Sample rate in samples/sec (default: 2e6)")
    parser.add_argument("--start_time", type=str, help="Start time in HH:MM format (24-hour)")
    args = parser.parse_args()

    detector = RealTime_OFDM_Detector(
        center_freq=args.freq,
        sample_rate=args.rate,
        gain=args.gain)
    detector.detect_ofdm()
