import sys
import time
import struct
import zlib
import logging
import numpy as np

logger = logging.getLogger(__name__)

def print_progress(iteration, total, start_time, prefix='', suffix='', decimals=1, length=50, fill='#'):
    """
    Call in a loop to create terminal progress bar
    """
    if total == 0:
        return
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    elapsed_time = time.time() - start_time
    if iteration > 0:
        eta = elapsed_time * (total / iteration - 1)
        eta_str = f"{eta:.1f}s"
    else:
        eta_str = "?"

    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix} ETA: {eta_str}')
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

class OFDMProtocol:
    def __init__(self, n_fft=64, n_cp=16, pilot_interval=8, sample_rate=12000, carrier_freq=12000, min_freq=0, max_freq=48000):
        logger.debug(f"OFDMProtocol initialized with n_fft={n_fft}, n_cp={n_cp}, pilot_interval={pilot_interval}, sample_rate={sample_rate}, carrier={carrier_freq}, range=[{min_freq}, {max_freq}]")
        self.n_fft = n_fft
        self.n_cp = n_cp
        self.pilot_interval = pilot_interval
        self.carrier_freq = carrier_freq
        
        # Subcarrier allocation
        self.sc_all = np.arange(n_fft)
        # Null DC (0) and Nyquist (n_fft/2)
        self.sc_null = np.array([0, n_fft//2])

        # Apply frequency constraints
        df = sample_rate / n_fft
        freq_nulls = []
        for i in range(n_fft):
            f_rel = i * df
            if i >= n_fft // 2:
                f_rel -= sample_rate
            f_abs = carrier_freq + f_rel
            
            if f_abs < min_freq or f_abs > max_freq:
                freq_nulls.append(i)
        
        if freq_nulls:
            self.sc_null = np.unique(np.concatenate((self.sc_null, np.array(freq_nulls, dtype=int))))

        # Pilots (inserted at fixed intervals)
        self.sc_pilot = self.sc_all[1::pilot_interval]
        # Remove pilots that fall on null subcarriers (e.g. Nyquist)
        self.sc_pilot = np.setdiff1d(self.sc_pilot, self.sc_null)
        # Data (remaining subcarriers)
        mask = np.ones(n_fft, dtype=bool)
        mask[self.sc_null] = False
        mask[self.sc_pilot] = False
        self.sc_data = self.sc_all[mask]
        
        self.n_data = len(self.sc_data)
        
        if self.n_data == 0:
             raise ValueError("No data subcarriers available with current frequency constraints.")
        
        # Generate Preamble for synchronization (Random BPSK sequence)
        # Using a fixed seed ensures the receiver knows the preamble structure
        rng = np.random.RandomState(42)
        preamble_freq = np.zeros(n_fft, dtype=complex)
        # Use valid subcarriers for preamble (avoiding DC/Nyquist)
        idx = np.arange(1, n_fft)
        idx = idx[idx != n_fft//2]
        preamble_freq[idx] = rng.choice([-1, 1], size=len(idx))
        
        # Time-domain preamble, scaled to match data symbols
        self.preamble_t = np.fft.ifft(preamble_freq) * self.n_fft

        if logger.isEnabledFor(logging.DEBUG):
            self._log_configuration(sample_rate)

    def _log_configuration(self, fs):
        df = fs / self.n_fft
        t_fft = (self.n_fft / fs) * 1000 # ms
        t_cp = (self.n_cp / fs) * 1000 # ms
        t_total = t_fft + t_cp
        
        logger.debug(f"--- OFDM Configuration Dump (Baseband Fs={fs}Hz) ---")
        logger.debug(f"FFT Size: {self.n_fft} bins")
        logger.debug(f"Cyclic Prefix: {self.n_cp} samples")
        logger.debug(f"Subcarrier Spacing: {df:.2f} Hz")
        logger.debug(f"Symbol Duration: {t_total:.3f} ms (Useful: {t_fft:.3f} ms, CP: {t_cp:.3f} ms)")
        logger.debug(f"Data Subcarriers: {len(self.sc_data)}")
        logger.debug(f"Pilot Subcarriers: {len(self.sc_pilot)}")
        logger.debug(f"Null Subcarriers: {len(self.sc_null)}")
        
        logger.debug("--- Frequency Map (Baseband + Absolute) ---")
        for i in range(self.n_fft):
            freq = i * df
            if i > self.n_fft // 2: freq -= fs
            f_abs = self.carrier_freq + freq
            
            role = "DATA"
            if i in self.sc_null: role = "NULL"
            elif i in self.sc_pilot: role = "PILOT"
            
            logger.debug(f"Bin {i:3d}: {freq:>10.1f} Hz (Abs: {f_abs:>10.1f} Hz) | {role}")
        logger.debug("-------------------------------")

    def bpsk_mod(self, bits):
        """Maps bits to BPSK symbols."""
        # Map 0 -> -1, 1 -> 1
        return (2*bits - 1) + 0j

    def bpsk_demod(self, symbols):
        """Demaps BPSK symbols to bits."""
        return (symbols.real > 0).astype(int)

    def qpsk_mod(self, bits):
        """Maps bits to QPSK symbols."""
        # 00 -> -1-1j, 01 -> -1+1j, 10 -> 1-1j, 11 -> 1+1j
        if len(bits) % 2 != 0:
            raise ValueError("Bits length must be even for QPSK")
        
        b0 = bits[0::2]
        b1 = bits[1::2]
        # Map 0->-1, 1->1
        return (2*b0 - 1) + 1j*(2*b1 - 1)

    def qpsk_demod(self, symbols):
        """Demaps QPSK symbols to bits."""
        b0 = (symbols.real > 0).astype(int)
        b1 = (symbols.imag > 0).astype(int)
        bits = np.zeros(len(symbols) * 2, dtype=int)
        bits[0::2] = b0
        bits[1::2] = b1
        return bits

    def ask_mod(self, bits):
        """Maps bits to ASK symbols (0 -> 0, 1 -> 1)."""
        return bits.astype(complex)

    def ask_demod(self, symbols):
        """Demaps ASK symbols to bits based on magnitude."""
        return (np.abs(symbols) > 0.5).astype(int)

    def transmit(self, payload_bits, modulation='qpsk'):
        """Encodes bits into an OFDM time-domain signal."""
        # 1. Pad payload to fit integer number of OFDM symbols
        if modulation == 'qpsk':
            bits_per_symbol = self.n_data * 2 # 2 bits per symbol (QPSK)
            mod_func = self.qpsk_mod
        elif modulation == 'bpsk':
            bits_per_symbol = self.n_data * 1 # 1 bit per symbol (BPSK)
            mod_func = self.bpsk_mod
        elif modulation == 'ask':
            bits_per_symbol = self.n_data * 1 # 1 bit per symbol (ASK)
            mod_func = self.ask_mod
        else:
            raise ValueError(f"Unknown modulation scheme: {modulation}")

        n_syms = int(np.ceil(len(payload_bits) / bits_per_symbol))
        n_pad = n_syms * bits_per_symbol - len(payload_bits)
        
        if n_pad > 0:
            bits = np.concatenate([payload_bits, np.zeros(n_pad, dtype=int)])
        else:
            bits = payload_bits
        
        # 2. Modulate Data
        symbols = mod_func(bits)
        
        # 3. Construct OFDM Frame
        signal_list = []
        
        # --- Header Generation & Preamble Repetition ---
        # We repeat the Preamble + Header 3 times for robustness.
        # Header Format: Magic(1B) | Rep_Idx(1B) | Length(2B) | CRC(1B) = 5 Bytes
        header_magic = 0xAC
        
        for rep_idx in range(3):
            header_bytes = struct.pack('>BBH', header_magic, rep_idx, n_syms)
            header_crc = zlib.crc32(header_bytes) & 0xFF
            header_bytes += struct.pack('B', header_crc)
            
            header_bits = np.unpackbits(np.frombuffer(header_bytes, dtype=np.uint8))
            # Pad to fill one BPSK symbol
            if len(header_bits) > self.n_data:
                 raise ValueError(f"FFT size too small for header. Need {len(header_bits)} bits, have {self.n_data}")
            
            header_bits_padded = np.concatenate([header_bits, np.zeros(self.n_data - len(header_bits), dtype=int)])
            if modulation == 'ask':
                header_symbol = self.ask_mod(header_bits_padded)
            else:
                header_symbol = self.bpsk_mod(header_bits_padded)

            # Add Preamble
            preamble_cp = self.preamble_t[-self.n_cp:]
            preamble_with_cp = np.concatenate([preamble_cp, self.preamble_t])
            signal_list.append(preamble_with_cp)
            
            # Add Header
            freq_header = np.zeros(self.n_fft, dtype=complex)
            freq_header[self.sc_data] = header_symbol
            freq_header[self.sc_pilot] = 1+0j
            time_header = np.fft.ifft(freq_header)
            time_header *= self.n_fft # Boost
            header_cp = time_header[-self.n_cp:]
            header_with_cp = np.concatenate([header_cp, time_header])
            signal_list.append(header_with_cp)
        
        # Process Data Symbols
        data_per_sym = self.n_data


        start_time = time.time()
        for i in range(n_syms):
            chunk = symbols[i*data_per_sym : (i+1)*data_per_sym]
            
            # Map to frequency domain
            freq_data = np.zeros(self.n_fft, dtype=complex)
            freq_data[self.sc_data] = chunk
            freq_data[self.sc_pilot] = 1+0j # Known pilot value (1+0j)
            
            # IFFT to Time Domain
            time_data = np.fft.ifft(freq_data)
            
            # Boost data symbol power to match preamble (compensate for 1/N IFFT scaling)
            time_data *= self.n_fft
            
            # Add Cyclic Prefix
            cp = time_data[-self.n_cp:]
            symbol = np.concatenate([cp, time_data])
            
            signal_list.append(symbol)
            
            if i % 100 == 0 or i == n_syms - 1:
                print_progress(i + 1, n_syms, start_time, prefix='OFDM Mod')
            
        return np.concatenate(signal_list)

    def receive(self, rx_signal, modulation='qpsk'):
        """Decodes an OFDM time-domain signal into bits, supporting multiple concatenated transmissions."""
        if modulation == 'qpsk':
            demod_func = self.qpsk_demod
        elif modulation == 'bpsk':
            demod_func = self.bpsk_demod
        elif modulation == 'ask':
            demod_func = self.ask_demod
        else:
            raise ValueError(f"Unknown modulation scheme: {modulation}")

        # 1. Synchronization: Find all potential preambles
        corr = np.correlate(rx_signal, self.preamble_t, mode='valid')
        peaks = np.abs(corr)
        
        if len(peaks) == 0:
            print("Warning: Correlation resulted in empty peaks array.")
            return np.array([], dtype=int)
            
        threshold = np.mean(peaks) + 4.0 * np.std(peaks)
        # Filter out cross-correlation sidelobes in high-SNR cases
        threshold = max(threshold, np.max(peaks) * 0.6)
        possible_peaks = np.where(peaks > threshold)[0]

        if len(possible_peaks) == 0:
            peak_idx = np.argmax(peaks)
            if peaks[peak_idx] > np.mean(peaks): # Basic sanity check
                 possible_peaks = np.array([peak_idx])
            else:
                print("Warning: No preamble found above noise floor.")
                return np.array([], dtype=int)

        # 2. Debounce peaks to find unique frame starts
        min_dist = len(self.preamble_t) # Minimum distance between preambles
        
        # Sort by peak strength to prioritize stronger signals
        sorted_peak_indices = possible_peaks[np.argsort(peaks[possible_peaks])[::-1]]
        
        debounced_peaks = []
        for p_idx in sorted_peak_indices:
            is_close = False
            for dp_idx in debounced_peaks:
                if abs(p_idx - dp_idx) < min_dist:
                    is_close = True
                    break
            if not is_close:
                debounced_peaks.append(p_idx)
        
        debounced_peaks.sort()
        print(f"Found {len(debounced_peaks)} potential transmission(s).")

        # 3. Decode each transmission segment
        all_bits_list = []
        symbol_len = self.n_fft + self.n_cp
        processed_upto = 0

        for i, frame_start in enumerate(debounced_peaks):
            if frame_start < processed_upto:
                continue

            # Attempt to decode Header (immediately follows preamble)
            header_idx = frame_start + self.n_fft
            if header_idx + symbol_len > len(rx_signal):
                continue

            # Decode Header Symbol
            sym_time = rx_signal[header_idx + self.n_cp : header_idx + symbol_len]
            sym_freq = np.fft.fft(sym_time)
            
            # Equalize Header
            pilots_rx = sym_freq[self.sc_pilot]
            with np.errstate(divide='ignore', invalid='ignore'):
                h_est = (pilots_rx / (1+0j))
            h_est_full = np.interp(self.sc_all, self.sc_pilot, h_est.real) + 1j*np.interp(self.sc_all, self.sc_pilot, h_est.imag)
            
            sym_eq = np.nan_to_num(sym_freq / h_est_full)
            if modulation == 'ask':
                header_bits_raw = self.ask_demod(sym_eq[self.sc_data])
            else:
                header_bits_raw = self.bpsk_demod(sym_eq[self.sc_data])
            
            # Check Header CRC (Now 5 bytes: Magic, Rep, Len, CRC)
            header_bytes = np.packbits(header_bits_raw[:40]).tobytes()
            try:
                magic, rep_idx, length, crc = struct.unpack('>BBHB', header_bytes)
                calc_crc = zlib.crc32(header_bytes[:4]) & 0xFF
                if magic != 0xAC or crc != calc_crc:
                    # print(f"Skipping false positive at {frame_start} (Bad Header)")
                    continue
                if rep_idx > 2:
                    continue
            except Exception:
                continue

            print(f"Valid Header (Rep {rep_idx+1}/3) at sample {frame_start}: {length} symbols.")
            
            # Start decoding Data
            # Skip remaining repetitions (each rep is Preamble + Header)
            len_ph = symbol_len * 2
            reps_remaining = 2 - rep_idx
            current_idx = header_idx + symbol_len + (reps_remaining * len_ph)
            
            # Calculate exact end based on header length
            next_boundary = current_idx + (length * symbol_len)
            
            if next_boundary > len(rx_signal):
                print("Warning: Signal truncated based on header length.")
                next_boundary = len(rx_signal)
            
            rx_bits_list = []
            total_symbols = (next_boundary - current_idx) // symbol_len
            processed_symbols = 0
            start_time = time.time()
            
            # Process symbols until we hit the next boundary
            while current_idx + symbol_len <= next_boundary:
                sym_time_cp = rx_signal[current_idx : current_idx + symbol_len]
                sym_time = sym_time_cp[self.n_cp:]
                sym_freq = np.fft.fft(sym_time)
                
                pilots_rx = sym_freq[self.sc_pilot]
                pilots_tx = 1+0j
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    h_est_pilots = pilots_rx / pilots_tx
                
                h_est_real = np.interp(self.sc_all, self.sc_pilot, h_est_pilots.real)
                h_est_imag = np.interp(self.sc_all, self.sc_pilot, h_est_pilots.imag)
                h_est = h_est_real + 1j*h_est_imag
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    sym_eq = sym_freq / h_est
                sym_eq = np.nan_to_num(sym_eq)
                
                data_syms = sym_eq[self.sc_data]
                bits = demod_func(data_syms)
                rx_bits_list.append(bits)
                
                current_idx += symbol_len
                processed_symbols += 1
                if processed_symbols % 100 == 0 or processed_symbols == total_symbols:
                    print_progress(processed_symbols, total_symbols, start_time, prefix='OFDM Demod')
            
            if rx_bits_list:
                all_bits_list.append(np.concatenate(rx_bits_list))
                
            processed_upto = next_boundary

        if all_bits_list:
            return np.concatenate(all_bits_list)
        else:
            return np.array([], dtype=int)
