#!/usr/bin/env python
# coding: utf-8
# PURPOSE: OFDM RealTime Signal Transmitter
# Aug. 2025

import uhd
import numpy as np
import time
import sys
import signal
import argparse
import importlib
import requests
from requests.auth import HTTPBasicAuth
import csv
import datetime
# Reload the watermark generator module
import OFDM_signal_generator
importlib.reload(OFDM_signal_generator)
from OFDM_signal_generator import OFDM_Generator

class OFDM_TX:
    def adjust_parameters_by_sample_rate(self):
        base_rate = 2e6  # reference sampling rate
        scale = self.samp_rate / base_rate
        self.chip_samples = int(640 * scale)
        self.FFT = int(64 * scale)
        self.CP = int(16 * scale)
        self.OFDM_size = self.FFT + self.CP
        # print(f"[Adjusted TX Parameters] chip_samples: {self.chip_samples}, FFT: {self.FFT}, CP: {self.CP}, OFDM_size: {self.OFDM_size}")

    lo_adjust = 1.5e6  # LO offset adjustment
    master_clock = 200e6  # Master clock rate

    def __init__(self, addr="192.168.40.2", external_clock=False, chan=0,
                 center_freq=3385e6, gain=30, samp_rate=2e6, repeat=10, start_epoch=None):
        # Initialize SDR and transmission parameters
        self.addr = addr
        self.external_clock = external_clock
        self.channel = chan
        self.center_freq = center_freq
        self.gain = gain
        self.samp_rate = samp_rate
        self.adjust_parameters_by_sample_rate()
        self.usrp = None
        self.txstreamer = None
        self.keep_running = True
        self.generator = OFDM_Generator()
        self.start_epoch = start_epoch
        self.tx_repeat = repeat
        
    def init_radio(self):
        # Initialize USRP device
        self.usrp = uhd.usrp.MultiUSRP(f"addr={self.addr}")
        if self.external_clock:
            self.usrp.set_time_source("external")
            self.usrp.set_clock_source("external")
        self.usrp.set_master_clock_rate(self.master_clock)
        self.usrp.set_tx_antenna("TX/RX", self.channel)

    def setup_streamers(self):
        # Setup TX streamer
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [self.channel]
        self.txstreamer = self.usrp.get_tx_stream(st_args)

    def tune(self, freq, gain, rate, use_lo_offset=True):
        # Configure radio parameters
        self.currate = rate
        self.usrp.set_tx_rate(rate, self.channel)
        if use_lo_offset: 
            lo_off = rate / 2 + self.lo_adjust
            tune_req = uhd.types.TuneRequest(freq, lo_off)
        else:
            tune_req = uhd.types.TuneRequest(freq)
        self.usrp.set_tx_freq(tune_req, self.channel)
        self.usrp.set_tx_gain(gain, self.channel)

    def Set_all_params(self):
        # Apply all configuration steps
        self.init_radio()
        self.setup_streamers()
        self.tune(self.center_freq, self.gain, self.samp_rate)

    def send_samples(self, samples):
        # Transmit samples over SDR
        meta = uhd.types.TXMetadata()
        meta.start_of_burst = True
        meta.end_of_burst = False

        max_samps = self.txstreamer.get_max_num_samps() - 4
        # print('Max transmit buffer:', max_samps)
        total = samples.size
        idx = 0

        total_requested = 0
        total_sent = 0
        drop_events = 0

        while idx < total:
            nsamps = min(total - idx, max_samps)
            buf = np.zeros((1, max_samps), dtype=np.complex64)
            buf[0, :nsamps] = samples[idx:idx + nsamps]

            if idx + nsamps >= total:
                meta.end_of_burst = True

            sent = self.txstreamer.send(buf, meta)

            total_requested += nsamps
            total_sent += sent

            if sent == 0:
                drop_events += 1
                print(f"[WARNING] {time.strftime('%Y-%m-%d %H:%M:%S')} - Drop detected.")
            elif sent < nsamps:
                print(f"[INFO] {time.strftime('%Y-%m-%d %H:%M:%S')} - Partial send: {sent}/{nsamps}")

            idx += sent

        print(f"[TX SUMMARY] Sent {total_sent}/{total_requested} samples. Drop events: {drop_events}")

    def run(self):

        if self.start_epoch is not None:
            print(f"[INFO] Waiting until {self.start_epoch} to start transmission...")
            while time.time() < self.start_epoch:
                if not self.keep_running:
                    print("[INFO] Transmission aborted during wait.")
                    return
                time.sleep(0.01)
            print("[INFO] Starting transmission.")
            
        # Main loop to transmit pseudonyms and log detection intervals
        elapsed_times = []
        last_detection_time = None
  
        send_start_time = time.time()
        tx_signal =  self.generator.generate_ofdm_packet()
        print('Length of tx_signal', len(tx_signal))
        iteration = 0
        while self.keep_running:
            
            self.send_samples(tx_signal)
            print('Transmitting OFDM Samples')
            time.sleep(0.1)
            iteration +=1
            # if iteration == self.tx_repeat:
            #     self.keep_running = False        

def handle_interrupt(sig, frame):
    print("\n[INFO] Graceful shutdown triggered.")
    tx.keep_running = False

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watermark TX RealTime")
    parser.add_argument("-f", "--freq", type=float, default=3385e6, help="Center frequency in Hz (default: 3385e6)")
    parser.add_argument("-g", "--gain", type=float, default=30, help="Transmit gain in dB (default: 30)")
    parser.add_argument("-r", "--rate", type=float, default=2e6, help="Sample rate in samples/sec (default: 2e6)")
    parser.add_argument("--start_time", type=str, help="Start time in HH:MM format (24-hour)")
    args = parser.parse_args()

    start_epoch = None
    if args.start_time:
        now = datetime.datetime.now()
        hh, mm = map(int, args.start_time.split(":"))
        start_dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if start_dt < now:
            start_dt += datetime.timedelta(days=1)
        start_epoch = start_dt.timestamp()

    tx = OFDM_TX(center_freq=args.freq, gain=args.gain, samp_rate=args.rate, start_epoch=start_epoch)
    signal.signal(signal.SIGINT, handle_interrupt)
    tx.Set_all_params()
    tx.run()
