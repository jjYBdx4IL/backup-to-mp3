import argparse
import os
import wave
import tempfile
import subprocess
import logging
import numpy as np
from scipy.io import wavfile
from scipy import signal as sp_signal

from ofdm_protocol import OFDMProtocol
from file_xfer_protocol import FileTransferProtocol
from model_impulse_response import apply_ir


def parse_n_cp(n_cp_arg, baseband_sample_rate):
    """Parses the n_cp argument which can be an integer, 'mp3', or 'Xms'."""
    if isinstance(n_cp_arg, (int, float)):
        return int(n_cp_arg)
    
    s = str(n_cp_arg).lower().strip()
    
    if s == 'mp3':
        # 26.1 ms window to cover MP3 smearing/pre-echo
        return int(np.ceil(0.0261 * baseband_sample_rate))
    elif s.endswith('ms'):
        try:
            ms = float(s[:-2])
            return int(np.ceil(ms / 1000.0 * baseband_sample_rate))
        except ValueError:
             raise ValueError(f"Invalid format for n_cp: {n_cp_arg}")
    
    try:
        return int(s)
    except ValueError:
         raise ValueError(f"Invalid format for n_cp: {n_cp_arg}")

def run_simulation(packet_size=32, parity_pct=25, modulation='qpsk', carrier_freq=12000,
                   multipath=False, noise_power=0.0, disruption_pct=0.0, delay=0, data_len=None,
                   input_path=None, output_path=None, mp3_bitrate=None, debug=False, repeat=1,
                   n_fft=64, n_cp=16, pilot_interval=8, min_freq=0, max_freq=24000, ir_path=None,
                   adaptive_normalization=False):
    """Runs the file transfer simulation with configurable channel distortions."""
    print(f"Configuration: FFT={n_fft}, CP={n_cp}, Pilot={pilot_interval}, Mod={modulation}, Parity={parity_pct}%, PktSize={packet_size}, Carrier={carrier_freq}Hz")
    print(f"Frequency Constraints: {min_freq} Hz - {max_freq} Hz")
    if mp3_bitrate:
        print(f"MP3 Simulation: Bitrate={mp3_bitrate}")
    print(f"Channel: Multipath={multipath}, Noise={noise_power}, Disruption={disruption_pct}%, Delay={delay}")
    
    # 1. Prepare Input Data
    temp_input = False
    if input_path:
        source_file = input_path
    else:
        temp_input = True
        fd_in, source_file = tempfile.mkstemp()
        os.close(fd_in)
        
        if data_len:
            rng = np.random.RandomState(42)
            original_msg = bytes(rng.randint(0, 256, data_len, dtype=np.uint8))
        else:
            original_msg = b"Hello World! This is a robust file transfer over OFDM with Reed-Solomon FEC." * 50
        
        with open(source_file, 'wb') as f:
            f.write(original_msg)
        print(f"Generated {len(original_msg)} bytes of test data.")
    
    with open(source_file, 'rb') as f:
        sent_data = f.read()
    print(f"Original Message ({len(sent_data)} bytes):\n{sent_data[:50]}...")

    # 2. Transmit (Send File)
    fd_tx, tx_wav_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd_tx)
    
    print("Transmitting (Generating WAV)...")
    send_file(source_file, tx_wav_path, n_fft=n_fft, n_cp=n_cp, pilot_interval=pilot_interval,
              modulation=modulation, parity_pct=parity_pct, packet_size=packet_size,
              carrier_freq=carrier_freq, repeat=repeat, min_freq=min_freq, max_freq=max_freq)

    # 3. Channel Simulation
    # We need to modify the WAV file if there are channel effects
    if multipath or noise_power > 0 or disruption_pct > 0 or delay > 0 or mp3_bitrate:
        print("Applying Channel Effects...")
        
        # Read the generated WAV
        fs, audio_data = wavfile.read(tx_wav_path)
        if len(audio_data.shape) > 1: audio_data = audio_data[:, 0]
        
        # Normalize to float for processing
        signal_float = audio_data.astype(np.float32) / 32768.0
        
        # MP3 Simulation
        if mp3_bitrate:
            print(f"Simulating MP3 compression at {mp3_bitrate}...")
            fd_mp3, temp_mp3 = tempfile.mkstemp(suffix='.mp3')
            os.close(fd_mp3)
            fd_dec, temp_decoded_wav = tempfile.mkstemp(suffix='.wav')
            os.close(fd_dec)
            
            ffmpeg_out = None if debug else subprocess.DEVNULL

            try:
                # Encode
                subprocess.run(["ffmpeg", "-y", "-i", tx_wav_path, "-b:a", mp3_bitrate, temp_mp3], 
                               stdout=ffmpeg_out, stderr=ffmpeg_out, check=True)
                # Decode
                subprocess.run(["ffmpeg", "-y", "-i", temp_mp3, temp_decoded_wav], 
                               stdout=ffmpeg_out, stderr=ffmpeg_out, check=True)
                
                # Read back
                fs_mp3, data_mp3 = wavfile.read(temp_decoded_wav)
                if len(data_mp3.shape) > 1: data_mp3 = data_mp3[:, 0]
                signal_float = data_mp3.astype(np.float32) / 32768.0
            except Exception as e:
                print(f"MP3 simulation failed: {e}")
            finally:
                if os.path.exists(temp_mp3): os.remove(temp_mp3)
                if os.path.exists(temp_decoded_wav): os.remove(temp_decoded_wav)

        # Multipath (Real-valued echo for passband)
        if multipath:
            # Simple echo channel
            channel = np.array([1.0, 0.0, 0.0, 0.0, 0.4, 0.0, -0.1])
            signal_float = np.convolve(signal_float, channel, mode='full')

        # Noise
        if noise_power > 0:
            noise = np.random.randn(len(signal_float)) * np.sqrt(noise_power)
            signal_float += noise

        # Disruption
        if disruption_pct > 0:
            cut_start = len(signal_float) // 3
            cut_len = int(len(signal_float) * disruption_pct / 100.0)
            print(f"Simulating disruption: Zeroing out {cut_len} samples...")
            signal_float[cut_start : cut_start + cut_len] = 0

        # Delay
        if delay > 0:
            signal_float = np.concatenate([np.zeros(delay), signal_float])

        # Write back to tx_wav_path
        signal_float = np.clip(signal_float, -1.0, 1.0)
        audio_int16 = (signal_float * 32767).astype(np.int16)
        wavfile.write(tx_wav_path, fs, audio_int16)

    # Apply Impulse Response (Convolution) if requested
    if ir_path:
        print(f"Applying Impulse Response from: {ir_path}")
        fd_ir, temp_ir_wav = tempfile.mkstemp(suffix='.wav')
        os.close(fd_ir)
        
        class IRArgs:
            def __init__(self, input_path, ir_path, output_path):
                self.input = input_path
                self.impulse_response = ir_path
                self.output = output_path
        
        try:
            apply_ir(IRArgs(tx_wav_path, ir_path, temp_ir_wav))
            
            # Swap paths: temp_ir_wav becomes the new tx_wav_path for reception
            if os.path.exists(tx_wav_path):
                os.remove(tx_wav_path)
            tx_wav_path = temp_ir_wav
        except Exception as e:
            print(f"Failed to apply Impulse Response: {e}")
    
    # 4. Receive (Recv File)
    temp_output = False
    if output_path:
        sink_file = output_path
    else:
        temp_output = True
        fd_out, sink_file = tempfile.mkstemp()
        os.close(fd_out)

    print("Receiving (Decoding WAV)...")
    recv_file(tx_wav_path, sink_file, n_fft=n_fft, n_cp=n_cp, pilot_interval=pilot_interval,
              modulation=modulation, parity_pct=parity_pct, packet_size=packet_size,
              carrier_freq=carrier_freq, min_freq=min_freq, max_freq=max_freq,
              adaptive_normalization=adaptive_normalization)

    # 5. Verification
    print("Verifying...")
    try:
        with open(sink_file, 'rb') as f:
            rec_data = f.read()
    except FileNotFoundError:
        rec_data = b""

    print(f"Sent: {len(sent_data)} bytes")
    print(f"Received: {len(rec_data)} bytes")
    
    if sent_data == rec_data:
        print("SUCCESS: File recovered perfectly despite channel errors!")
    else:
        min_len = min(len(sent_data), len(rec_data))
        diffs = sum(1 for a, b in zip(sent_data[:min_len], rec_data[:min_len]) if a != b)
        diffs += abs(len(sent_data) - len(rec_data))
        print(f"FAILURE: {diffs} byte errors.")

    # Calculate effective data rate
    try:
        with wave.open(tx_wav_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            if duration > 0:
                kbps = (len(sent_data) * 8) / 1000.0 / duration
                print(f"Effective Data Rate: {kbps:.2f} kbits/s (Payload: {len(sent_data)} bytes in {duration:.2f}s)")
    except Exception as e:
        print(f"Could not calculate data rate: {e}")

    # Cleanup
    if os.path.exists(tx_wav_path):
        os.remove(tx_wav_path)
    if temp_input and os.path.exists(source_file):
        os.remove(source_file)
    if temp_output and os.path.exists(sink_file):
        os.remove(sink_file)

def _get_tuning_layout(ofdm, known_bits):
    """Returns a dictionary with the layout of the tuning signal."""
    preamble_len = ofdm.n_fft + ofdm.n_cp
    symbol_len = ofdm.n_fft + ofdm.n_cp
    
    # BPSK block
    n_syms_bpsk = int(np.ceil(len(known_bits) / (ofdm.n_data * 1)))
    bpsk_block_len = n_syms_bpsk * symbol_len
    
    # CFO block
    cfo_syms = 16
    cfo_block_len = cfo_syms * symbol_len
    
    # QPSK block
    n_syms_qpsk = int(np.ceil(len(known_bits) / (ofdm.n_data * 2)))
    qpsk_block_len = n_syms_qpsk * symbol_len
    
    return {
        "preamble": preamble_len,
        "bpsk_data": bpsk_block_len,
        "cfo_tones": cfo_block_len,
        "qpsk_data": qpsk_block_len,
        "known_bits": known_bits
    }

def generate_tuning_file(output_path, n_fft=64, n_cp=16, carrier_freq=12000):
    """Generates a WAV file with a sequence of modulated signals for channel tuning."""
    print(f"Generating tuning file at: {output_path}")
    
    baseband_sample_rate = 12000 # 48000 / 4
    n_cp_int = parse_n_cp(n_cp, baseband_sample_rate)
    ofdm = OFDMProtocol(n_fft=n_fft, n_cp=n_cp_int)
    
    # Use a repeating pattern for easy verification
    known_bits = np.tile([1, 0, 1, 0, 1, 1, 0, 0], 128) # 1024 bits
    layout = _get_tuning_layout(ofdm, known_bits)

    # --- Construct Signal Blocks ---
    print("Creating signal blocks (Tones, BPSK, QPSK)...")
    
    # 1. BPSK modulated data (includes the preamble for sync)
    bpsk_signal = ofdm.transmit(known_bits, modulation='bpsk')
    
    # 2. Pure Tones for CFO/Phase estimation
    freq_tones = np.zeros(ofdm.n_fft, dtype=complex)
    freq_tones[ofdm.sc_pilot] = 1.0
    time_tones = np.fft.ifft(freq_tones)
    cp_tones = time_tones[-ofdm.n_cp:]
    symbol_tones = np.concatenate([cp_tones, time_tones])
    cfo_block = np.tile(symbol_tones, 16)

    # 3. QPSK modulated data (raw symbols, no preamble)
    qpsk_signal_full = ofdm.transmit(known_bits, modulation='qpsk')
    qpsk_signal_no_preamble = qpsk_signal_full[layout["preamble"]:]

    # --- Concatenate all blocks ---
    final_signal = np.concatenate([
        bpsk_signal,
        cfo_block,
        qpsk_signal_no_preamble
    ])

    # --- Upsample baseband signal to fit in audio bandwidth ---
    upsample_factor = 4
    final_signal = sp_signal.resample(final_signal, len(final_signal) * upsample_factor)
    
    # --- Convert to Audio WAV file ---
    print("Converting to mono WAV file...")
    sample_rate = 48000
    t = np.arange(len(final_signal)) / sample_rate
    carrier_cos = np.cos(2 * np.pi * carrier_freq * t)
    carrier_sin = np.sin(2 * np.pi * carrier_freq * t)
    mono_signal = final_signal.real * carrier_cos - final_signal.imag * carrier_sin
    
    max_amp = np.max(np.abs(mono_signal))
    if max_amp == 0: max_amp = 1
    scaling_factor = 32767 / max_amp
    mono_int16 = (mono_signal * scaling_factor).astype(np.int16)

    wavfile.write(output_path, sample_rate, mono_int16)
    
    print(f"Success! Tuning file '{output_path}' written.")
    print(f"Signal is Mono, modulated onto a {carrier_freq/1000:.1f} kHz carrier.")

def analyze_tuning_file(input_path, n_fft=64, n_cp=16, carrier_freq=12000):
    """Analyzes a received tuning WAV file to determine channel characteristics."""
    print(f"--- Analyzing: {input_path} ---")
    
    # 1. Read and Downconvert Signal
    try:
        sample_rate, data = wavfile.read(input_path)
        if len(data.shape) > 1:
            print("Warning: Expected mono audio, got stereo. Using left channel.")
            data = data[:, 0]
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return

    # Normalize to float
    rx_passband = data.astype(float) / 32768.0
    
    t = np.arange(len(rx_passband)) / sample_rate
    carrier_cos = np.cos(2 * np.pi * carrier_freq * t)
    carrier_sin = np.sin(2 * np.pi * carrier_freq * t)
    
    i_comp = rx_passband * 2 * carrier_cos
    q_comp = rx_passband * -2 * carrier_sin
    
    # Decimate combines a low-pass filter and downsampling.
    # Using an FIR filter for linear phase is important for I/Q data.
    upsample_factor = 4
    i_filt = sp_signal.decimate(i_comp, upsample_factor, ftype='fir')
    q_filt = sp_signal.decimate(q_comp, upsample_factor, ftype='fir')
    rx_signal = i_filt + 1j*q_filt

    baseband_sample_rate = sample_rate / upsample_factor
    n_cp_int = parse_n_cp(n_cp, baseband_sample_rate)
    ofdm = OFDMProtocol(n_fft=n_fft, n_cp=n_cp_int)

    # 2. Find Preamble
    corr = np.correlate(rx_signal, ofdm.preamble_t, mode='valid')
    peak_idx = np.argmax(np.abs(corr))
    frame_start = peak_idx
    print(f"Synchronization peak found at sample {frame_start}.")

    # 3. Slice a_nd Analyze Blocks
    known_bits = np.tile([1, 0, 1, 0, 1, 1, 0, 0], 128)
    layout = _get_tuning_layout(ofdm, known_bits)
    
    bpsk_start = frame_start + layout["preamble"]
    bpsk_end = bpsk_start + layout["bpsk_data"]
    cfo_start = bpsk_end
    cfo_end = cfo_start + layout["cfo_tones"]
    qpsk_start = cfo_end
    qpsk_end = qpsk_start + layout["qpsk_data"]
    
    # CFO Analysis
    cfo_block = rx_signal[cfo_start:cfo_end]
    symbol_len = n_fft + n_cp
    cfo_sym = cfo_block[:symbol_len][n_cp:] # First symbol, no CP
    cfo_freq = np.fft.fft(cfo_sym)
    
    # Compare received pilot locations to known ones
    pilot_pwr = np.abs(cfo_freq[ofdm.sc_pilot])
    noise_pwr = np.mean(np.abs(cfo_freq[~np.isin(ofdm.sc_all, ofdm.sc_pilot)]))
    snr_db = 10 * np.log10(np.mean(pilot_pwr) / noise_pwr)
    print(f"\nSNR Estimate: {snr_db:.2f} dB")

    # For now, CFO estimation is complex, so we'll skip the exact calculation
    print("CFO Estimate: (Not implemented)")

    # BER Analysis
    def analyze_ber(signal_block, mod_type, expected_bits):
        # A simplified demodulator loop for a block
        rx_bits_list = []
        for i in range(len(signal_block) // symbol_len):
            sym_cp = signal_block[i*symbol_len : (i+1)*symbol_len]
            sym = sym_cp[n_cp:]
            sym_f = np.fft.fft(sym)
            
            # Robust equalization (from main receive function)
            pilots_rx = sym_f[ofdm.sc_pilot]
            pilots_tx = 1.0

            with np.errstate(divide='ignore', invalid='ignore'):
                h_est_pilots = pilots_rx / pilots_tx

            h_est_real = np.interp(ofdm.sc_all, ofdm.sc_pilot, h_est_pilots.real)
            h_est_imag = np.interp(ofdm.sc_all, ofdm.sc_pilot, h_est_pilots.imag)
            h_est = h_est_real + 1j*h_est_imag
            
            with np.errstate(divide='ignore', invalid='ignore'):
                sym_f_eq = sym_f / h_est
            
            sym_f_eq = np.nan_to_num(sym_f_eq)
            sym_f_data = sym_f_eq[ofdm.sc_data]

            if mod_type == 'bpsk':
                bits = ofdm.bpsk_demod(sym_f_data)
            else: # qpsk
                bits = ofdm.qpsk_demod(sym_f_data)
            rx_bits_list.append(bits)
        
        rx_bits = np.concatenate(rx_bits_list)
        
        # Trim received bits to the maximum expected length first.
        # This handles the case where we demodulate more bits than expected due to symbol padding.
        if len(rx_bits) > len(expected_bits):
            rx_bits = rx_bits[:len(expected_bits)]
        
        # Now, compare against the portion of expected_bits that we actually received
        expected_bits_to_compare = expected_bits[:len(rx_bits)]

        errors = np.sum(rx_bits != expected_bits_to_compare)
        ber = errors / len(rx_bits) if len(rx_bits) > 0 else 0.0

        print(f"\n{mod_type.upper()} Analysis:")
        print(f"  - Recovered {len(rx_bits)} bits")
        print(f"  - Found {errors} errors")
        print(f"  - Bit Error Rate (BER): {ber:.4f}")

    analyze_ber(rx_signal[bpsk_start:bpsk_end], 'bpsk', known_bits)
    analyze_ber(rx_signal[qpsk_start:qpsk_end], 'qpsk', known_bits)
    print("\n--- Analysis Complete ---")

def send_file(input_path, output_path, n_fft=64, n_cp=16, pilot_interval=8, modulation='qpsk', parity_pct=25, packet_size=32, carrier_freq=12000, repeat=1, min_freq=0, max_freq=24000):
    """Encodes a file into a WAV signal using OFDM, writing to disk without high RAM usage."""
    print(f"Reading file: {input_path}")
    try:
        with open(input_path, 'rb') as f:
            file_data = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    audio_sample_rate = 48000
    upsample_factor = 4
    
    # Check for potential aliasing
    if carrier_freq < (audio_sample_rate / upsample_factor / 2):
        print(f"Warning: Carrier frequency {carrier_freq}Hz is likely too low for the signal bandwidth. Recommend > {audio_sample_rate/upsample_factor/2}Hz.")

    print(f"Opening WAV file for writing: {output_path}")
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(audio_sample_rate)

        n_cp_int = parse_n_cp(n_cp, audio_sample_rate/upsample_factor)

        for i in range(repeat):
            if repeat > 1:
                print(f"\n--- Generating and writing transmission {i+1}/{repeat} ---")
            
            print(f"Configuration: FFT={n_fft}, CP={n_cp_int} (req: {n_cp}), Pilot={pilot_interval}, Mod={modulation}, Parity={parity_pct}%, PktSize={packet_size}, Carrier={carrier_freq}Hz")
            
            ofdm = OFDMProtocol(n_fft=n_fft, n_cp=n_cp_int, pilot_interval=pilot_interval, 
                                sample_rate=audio_sample_rate/upsample_factor, carrier_freq=carrier_freq,
                                min_freq=min_freq, max_freq=max_freq)
            ftp = FileTransferProtocol(ofdm, packet_size=packet_size, parity_pct=parity_pct, modulation=modulation)

            # Generate baseband signal
            tx_signal = ftp.transmit_file(file_data, baseband_sample_rate=audio_sample_rate/upsample_factor)
            
            # Upsample and mix to carrier for this chunk
            tx_signal_up = sp_signal.resample(tx_signal, len(tx_signal) * upsample_factor)
            
            t = np.arange(len(tx_signal_up)) / audio_sample_rate
            carrier_cos = np.cos(2 * np.pi * carrier_freq * t)
            carrier_sin = np.sin(2 * np.pi * carrier_freq * t)
            mono_signal = tx_signal_up.real * carrier_cos - tx_signal_up.imag * carrier_sin
            
            max_amp = np.max(np.abs(mono_signal))
            if max_amp == 0: max_amp = 1
            scaling_factor = 32767 / max_amp
            mono_int16 = (mono_signal * scaling_factor).astype(np.int16)
            
            # Write this chunk's bytes to the WAV file
            wf.writeframes(mono_int16.tobytes())

    print(f"\nFinished writing {repeat} concatenated transmission(s) to: {output_path}")
    print("Done.")

def recv_file(input_path, output_path, n_fft=64, n_cp=16, pilot_interval=8, modulation='qpsk', parity_pct=25, packet_size=32, carrier_freq=12000, min_freq=0, max_freq=24000, adaptive_normalization=False):
    """Decodes a file from a WAV signal using OFDM."""
    print(f"Reading WAV: {input_path}")
    try:
        sample_rate, data = wavfile.read(input_path)
        if len(data.shape) > 1:
            print("Warning: Expected mono audio, got stereo. Using left channel.")
            data = data[:, 0]
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return

    # Normalize to float
    rx_passband = data.astype(float)
    if adaptive_normalization:
        peak = np.max(np.abs(rx_passband))
        if peak > 0:
            rx_passband /= peak
            print(f"Adaptive normalization applied. Peak amplitude: {peak}")
        else:
            rx_passband /= 32768.0
    else:
        rx_passband /= 32768.0
    
    # Downconvert
    t = np.arange(len(rx_passband)) / sample_rate
    carrier_cos = np.cos(2 * np.pi * carrier_freq * t)
    carrier_sin = np.sin(2 * np.pi * carrier_freq * t)
    
    i_comp = rx_passband * 2 * carrier_cos
    q_comp = rx_passband * -2 * carrier_sin
    
    upsample_factor = 4
    i_filt = sp_signal.decimate(i_comp, upsample_factor, ftype='fir')
    q_filt = sp_signal.decimate(q_comp, upsample_factor, ftype='fir')
    rx_signal = i_filt + 1j*q_filt
    
    n_cp_int = parse_n_cp(n_cp, sample_rate/upsample_factor)
    print(f"Configuration: FFT={n_fft}, CP={n_cp_int} (req: {n_cp}), Pilot={pilot_interval}, Mod={modulation}, Parity={parity_pct}%, PktSize={packet_size}, Carrier={carrier_freq}Hz")

    print("Decoding stream...")
    ofdm = OFDMProtocol(n_fft=n_fft, n_cp=n_cp_int, pilot_interval=pilot_interval, 
                        sample_rate=sample_rate/upsample_factor, carrier_freq=carrier_freq,
                        min_freq=min_freq, max_freq=max_freq)
    ftp = FileTransferProtocol(ofdm, packet_size=packet_size, parity_pct=parity_pct, modulation=modulation)
    
    recovered_data = ftp.receive_stream(rx_signal)
    
    if recovered_data:
        print(f"Recovered {len(recovered_data)} bytes.")
        try:
            with open(output_path, 'wb') as f:
                f.write(recovered_data)
            print(f"Saved to: {output_path}")
        except Exception as e:
            print(f"Error writing output file: {e}")
    else:
        print("Failed to recover any data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OFDM Protocol Simulation and Tuning Tool.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # Common arguments for file transfer
    ft_parser = argparse.ArgumentParser(add_help=False)
    ft_parser.add_argument('-m', '--modulation', type=str, default='ask', choices=['bpsk', 'qpsk', 'ask'], help='Modulation scheme (bpsk, qpsk, ask).')
    ft_parser.add_argument('--parity', type=float, default=25.0, help='Parity percentage (e.g. 25 for 25%% overhead).')
    ft_parser.add_argument('--packet-size', type=int, default=64, help='Packet size in bytes.')
    ft_parser.add_argument('--carrier', type=float, default=12000.0, help='Carrier frequency in Hz.')
    ft_parser.add_argument('--debug', action='store_true', help='Enable debug output (logs and ffmpeg).')
    ft_parser.add_argument('--n-fft', type=int, default=256, help='OFDM FFT size.')
    ft_parser.add_argument('--n-cp', type=str, default='57', help='OFDM cyclic prefix length (int, "mp3", or "Xms").')
    ft_parser.add_argument('--pilot-interval', type=int, default=8, help='OFDM pilot subcarrier interval.')
    ft_parser.add_argument('--min-freq', type=float, default=0.0, help='Minimum absolute frequency in Hz.')
    ft_parser.add_argument('--max-freq', type=float, default=16000.0, help='Maximum absolute frequency in Hz.')

    # Common arguments for tuning
    tune_parser = argparse.ArgumentParser(add_help=False)
    tune_parser.add_argument('--carrier', type=float, default=12000.0, help='Carrier frequency in Hz.')

    parser_sim = subparsers.add_parser('simulate', parents=[ft_parser], help='Run the built-in channel simulation.')
    parser_sim.add_argument('-i', '--input', type=str, help='Input file path (optional).')
    parser_sim.add_argument('-o', '--output', type=str, help='Output file path (optional).')
    parser_sim.add_argument('--multipath', action='store_true', help='Enable multipath distortion.')
    parser_sim.add_argument('--noise', type=float, default=0.0, help='Noise power (default 0.0).')
    parser_sim.add_argument('--disruption', type=float, default=0.0, help='Disruption percentage (default 0.0).')
    parser_sim.add_argument('--delay', type=int, default=0, help='Delay in samples (default 0).')
    parser_sim.add_argument('--data-len', type=int, default=None, help='Data length in bytes.')
    parser_sim.add_argument('--mp3-bitrate', type=str, help='MP3 bitrate (e.g. 64k) to simulate compression artifact.')
    parser_sim.add_argument('--repeat', type=int, default=1, help='Number of times to repeat the transmission.')
    parser_sim.add_argument('--ir', type=str, help='Path to Impulse Response WAV file to apply.')
    parser_sim.add_argument('--adaptive-normalization', action='store_true', help='Enable adaptive normalization of input signal.')
    
    parser_gen = subparsers.add_parser('generate', parents=[tune_parser], help='Generate a tuning WAV file.')
    parser_gen.add_argument('-o', '--output', type=str, required=True, help='Path to save the output tuning.wav file.')
    
    parser_an = subparsers.add_parser('analyze', parents=[tune_parser], help='Analyze a received tuning file.')
    parser_an.add_argument('-i', '--input', type=str, required=True, help='Path to the received tuning.wav file to analyze.')
    
    parser_tune = subparsers.add_parser('tune', parents=[tune_parser], help='Generate and analyze a loopback tuning file.')
    parser_tune.add_argument('-o', '--output', type=str, default='tuning_loopback.wav', help='Path for the temporary tuning file.')

    parser_send = subparsers.add_parser('send', parents=[ft_parser], help='Encode a file into a WAV signal.')
    parser_send.add_argument('-i', '--input', type=str, required=True, help='Input file path.')
    parser_send.add_argument('-o', '--output', type=str, required=True, help='Output WAV file path.')
    parser_send.add_argument('--repeat', type=int, default=1, help='Number of times to repeat the transmission.')

    parser_recv = subparsers.add_parser('recv', parents=[ft_parser], help='Decode a file from a WAV signal.')
    parser_recv.add_argument('-i', '--input', type=str, required=True, help='Input WAV file path.')
    parser_recv.add_argument('-o', '--output', type=str, required=True, help='Output file path.')
    parser_recv.add_argument('--adaptive-normalization', action='store_true', help='Enable adaptive normalization of input signal.')

    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO
    if hasattr(args, 'debug') and args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.command == 'generate':
        generate_tuning_file(args.output, carrier_freq=args.carrier)
    elif args.command == 'analyze':
        analyze_tuning_file(args.input, carrier_freq=args.carrier)
    elif args.command == 'simulate':
        run_simulation(packet_size=args.packet_size, parity_pct=args.parity, modulation=args.modulation, carrier_freq=args.carrier,
                       multipath=args.multipath, noise_power=args.noise, disruption_pct=args.disruption, delay=args.delay,
                       data_len=args.data_len, input_path=args.input, output_path=args.output,
                       mp3_bitrate=args.mp3_bitrate, debug=args.debug, repeat=args.repeat,
                       n_fft=args.n_fft, n_cp=args.n_cp, pilot_interval=args.pilot_interval,
                       min_freq=args.min_freq, max_freq=args.max_freq, ir_path=args.ir,
                       adaptive_normalization=args.adaptive_normalization)
    elif args.command == 'tune':
        print(f"Running loopback test. Using temporary file: {args.output}")
        generate_tuning_file(args.output, carrier_freq=args.carrier)
        analyze_tuning_file(args.output, carrier_freq=args.carrier)
    elif args.command == 'send':
        send_file(args.input, args.output, modulation=args.modulation, parity_pct=args.parity,
                  packet_size=args.packet_size, carrier_freq=args.carrier, repeat=args.repeat,
                  n_fft=args.n_fft, n_cp=args.n_cp, pilot_interval=args.pilot_interval,
                  min_freq=args.min_freq, max_freq=args.max_freq)
    elif args.command == 'recv':
        recv_file(args.input, args.output, modulation=args.modulation, parity_pct=args.parity,
                  packet_size=args.packet_size, carrier_freq=args.carrier,
                  n_fft=args.n_fft, n_cp=args.n_cp, pilot_interval=args.pilot_interval,
                  min_freq=args.min_freq, max_freq=args.max_freq,
                  adaptive_normalization=args.adaptive_normalization)
    else:
        parser.print_help()
