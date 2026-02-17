import math
import time
import struct
import zlib
import logging
import numpy as np
from reedsolo import RSCodec
from ofdm_protocol import print_progress

logger = logging.getLogger(__name__)

class FileTransferProtocol:
    def __init__(self, ofdm_proto, packet_size=64, parity_pct=20, modulation='qpsk'):
        logger.debug(f"FileTransferProtocol initialized with packet_size={packet_size}, parity_pct={parity_pct}, modulation={modulation}")
        self.ofdm = ofdm_proto
        self.packet_size = packet_size  # Payload size per packet
        self.parity_pct = parity_pct
        self.modulation = modulation
        
        self.rs = None # Will be initialized dynamically per block

        # Packet Header: SYNC(2) | SEQ(4) | TOTAL(4) | LEN(1) | CRC(4) = 15 bytes
        self.SYNC = b'\xAA\x55'
        self.HEADER_FMT = '>2sIIB' # SYNC, SEQ, TOTAL, LEN
        self.HEADER_LEN = struct.calcsize(self.HEADER_FMT)
        self.CRC_LEN = 4

    def _pack_packet(self, seq, total, payload):
        header = struct.pack(self.HEADER_FMT, self.SYNC, seq, total, len(payload))
        packet_content = header + payload
        crc = zlib.crc32(packet_content)
        return packet_content + struct.pack('>I', crc)

    def _calculate_block_structure(self, k_total):
        MAX_N = 255
        # Determine standard capacity per block
        k_std = math.floor(MAX_N / (1 + self.parity_pct / 100.0))
        
        if k_std < 1: k_std = 1
        
        # Number of blocks needed to fit k_total
        num_blocks = math.ceil(k_total / k_std)
        if num_blocks == 0: num_blocks = 1
        
        # Distribute k_total into num_blocks (Round Robin)
        base_k = k_total // num_blocks
        extra = k_total % num_blocks
        
        blocks = []
        for b in range(num_blocks):
            k_curr = base_k + (1 if b < extra else 0)
            
            # Calculate parity for this block
            p_curr = math.ceil(k_curr * self.parity_pct / 100.0)
            
            blocks.append((k_curr, p_curr))
            
        return blocks

    def transmit_file(self, data_bytes, baseband_sample_rate=12000.0):
        # Debug logging for transmission
        debug_data = data_bytes
        if len(debug_data) > 20:
             debug_data = f"<{len(debug_data)} bytes>"
        logger.debug(f"transmit_file called with data={debug_data}, baseband_sample_rate={baseband_sample_rate}")

        # 1. Fragment and RS Encode (Erasure Coding)
        # We stripe data across packets and interleave blocks to mitigate burst errors.
        
        # Prepend file length for exact recovery
        original_length = len(data_bytes)
        data_bytes = struct.pack('>I', original_length) + data_bytes
        
        # Calculate K (number of data packets needed)
        k_total = math.ceil(len(data_bytes) / self.packet_size)

        # Pad data to fill K packets exactly
        padded_len = k_total * self.packet_size
        data_bytes += b'\x00' * (padded_len - len(data_bytes))
        
        blocks = self._calculate_block_structure(k_total)
        print(f"Transmitting {k_total} data packets in {len(blocks)} interleaved blocks.")
        
        # Calculate Block Duration and Max Dropout
        if self.modulation == 'qpsk':
            bits_per_symbol = self.ofdm.n_data * 2
        elif self.modulation == 'bpsk' or self.modulation == 'ask':
            bits_per_symbol = self.ofdm.n_data * 1
        else:
            bits_per_symbol = self.ofdm.n_data * 2
            
        pkt_bits = (self.HEADER_LEN + self.packet_size + self.CRC_LEN) * 8
        syms_per_pkt = math.ceil(pkt_bits / bits_per_symbol)
        t_sym = (self.ofdm.n_fft + self.ofdm.n_cp) / baseband_sample_rate
        t_pkt = syms_per_pkt * t_sym
        
        if blocks:
            total_pkts = k_total + sum(p for k, p in blocks)
            t_total = total_pkts * t_pkt
            # Burst tolerance: approx num_blocks * avg_parity
            avg_p = sum(p for k, p in blocks) / len(blocks)
            burst_tol_pkts = len(blocks) * avg_p
            burst_tol_time = burst_tol_pkts * t_pkt
            print(f"Stats: Total Duration={t_total:.2f}s, Max Burst Tolerance=~{burst_tol_time:.4f}s ({int(burst_tol_pkts)} pkts)")
        
        # Prepare Data Payloads
        data_payloads = []
        for i in range(k_total):
            idx = i * self.packet_size
            data_payloads.append(bytearray(data_bytes[idx : idx + self.packet_size]))
            
        # Distribute to blocks for encoding (Round Robin)
        block_inputs = [[] for _ in range(len(blocks))]
        for i, payload in enumerate(data_payloads):
            block_inputs[i % len(blocks)].append(payload)
            
        parity_payloads_map = {} # block_idx -> list of parity payloads
        print("RS Encoding...")
        start_time = time.time()

        for b_idx, (k_curr, n_parity_curr) in enumerate(blocks):
            block_payloads = block_inputs[b_idx]
            # Add parity placeholders
            for _ in range(n_parity_curr):
                block_payloads.append(bytearray(self.packet_size))
            
            n_curr = k_curr + n_parity_curr
            
            if n_parity_curr > 0:
                rs = RSCodec(n_parity_curr)
                # Stripe-wise encoding for this block
                for i in range(self.packet_size):
                    column = bytearray()
                    for j in range(k_curr):
                        column.append(block_payloads[j][i])
                    
                    encoded_column = rs.encode(column)
                    
                    # The last n_parity_curr bytes are parity
                    for p in range(n_parity_curr):
                        block_payloads[k_curr + p][i] = encoded_column[k_curr + p]
            
            # Extract just the parity packets
            parity_payloads_map[b_idx] = block_payloads[k_curr:]
            
            if b_idx % 10 == 0:
                print_progress(b_idx + 1, len(blocks), start_time, prefix='RS Encode')
        
        print_progress(len(blocks), len(blocks), start_time, prefix='RS Encode')
        
        # Construct Final Packet List
        # 1. All Data Packets (in original sequence)
        final_payloads = list(data_payloads)
        
        # 2. Append Interleaved Parity Packets
        max_parity = max(p for k, p in blocks) if blocks else 0
        for p_idx in range(max_parity):
            for b_idx in range(len(blocks)):
                parity_list = parity_payloads_map[b_idx]
                if p_idx < len(parity_list):
                    final_payloads.append(parity_list[p_idx])

        # 2. Frame Packets
        tx_bits = np.array([], dtype=int)
        packet_list = []
        print("Framing Packets...")
        start_time = time.time()
        for seq, payload in enumerate(final_payloads):
            packet = self._pack_packet(seq, len(final_payloads), payload)
            packet_list.append(packet)
            if seq % 100 == 0 or seq == len(final_payloads) - 1:
                print_progress(seq + 1, len(final_payloads), start_time, prefix='Framing')
        all_bytes = b''.join(packet_list)
            
        # Convert to bits
        # We use numpy unpackbits for efficiency
        byte_arr = np.frombuffer(all_bytes, dtype=np.uint8)
        bits = np.unpackbits(byte_arr)
        
        # 3. Transmit via OFDM
        return self.ofdm.transmit(bits, modulation=self.modulation)

    def receive_stream(self, rx_signal):
        # 1. OFDM Demodulate
        print("Demodulating stream...")
        rx_bits = self.ofdm.receive(rx_signal, modulation=self.modulation)
        
        # 2. Convert bits to bytes
        # Ensure multiple of 8
        n_bits = len(rx_bits)
        rx_bits = rx_bits[:n_bits - (n_bits % 8)]
        rx_bytes = np.packbits(rx_bits).tobytes()
        
        # 3. Sliding Window Packet Search
        print("Searching for packets...")
        received_packets = {} # {seq: payload}
        valid_packets = set()
        total_packets = 0
        checksum_errors = 0
        
        start_time = time.time()
        total_bytes_to_scan = len(rx_bytes) - self.HEADER_LEN - self.CRC_LEN
        
        # Debug: Print first few bytes to verify we are receiving signal, not just noise
        # print(f"First 20 received bytes: {rx_bytes[:20].hex()}")
        last_progress_update = 0
        
        i = 0
        while i < len(rx_bytes) - self.HEADER_LEN - self.CRC_LEN:
            if i - last_progress_update > 10000:
                print_progress(i, total_bytes_to_scan, start_time, prefix='Scanning')
                last_progress_update = i
            
            # Check Sync
            if rx_bytes[i:i+2] == self.SYNC:
                try:
                    # Parse Header
                    _, seq, total, length = struct.unpack(self.HEADER_FMT, rx_bytes[i:i+self.HEADER_LEN])
                    
                    # Check bounds
                    if length > self.packet_size + 10: # Sanity check
                        i += 1
                        continue
                        
                    # Extract full packet for CRC check
                    pkt_len = self.HEADER_LEN + length
                    full_pkt_len = pkt_len + self.CRC_LEN
                    
                    if i + full_pkt_len > len(rx_bytes):
                        break # Not enough bytes left
                        
                    packet_content = rx_bytes[i : i+pkt_len]
                    received_crc = struct.unpack('>I', rx_bytes[i+pkt_len : i+full_pkt_len])[0]
                    
                    # Verify CRC
                    if zlib.crc32(packet_content) == received_crc:
                        # Valid Packet
                        payload = packet_content[self.HEADER_LEN:]
                        received_packets[seq] = payload
                        valid_packets.add(seq)
                        total_packets = total
                        i += full_pkt_len # Advance by full packet length
                        continue
                    else:
                        checksum_errors += 1
                        # CRC Failed - Retain faulty data but don't trust it fully
                        if seq not in valid_packets:
                            payload = packet_content[self.HEADER_LEN:]
                            # Ensure payload size matches expected packet size to avoid IndexError later
                            if len(payload) != self.packet_size:
                                if len(payload) > self.packet_size: payload = payload[:self.packet_size]
                                else: payload = payload + b'\x00' * (self.packet_size - len(payload))
                            received_packets[seq] = payload
                        i += 1 # Advance past the bad sync marker
                        continue
                except Exception:
                    # Header parsing failed, advance past the bad sync marker
                    i += 1
                    continue
            
            # No sync marker found, advance
            i += 1
        print_progress(total_bytes_to_scan, total_bytes_to_scan, start_time, prefix='Scanning')

        total_found_packets = len(valid_packets) + checksum_errors
        error_rate = (checksum_errors / total_found_packets * 100) if total_found_packets > 0 else 0.0
        print(f"Checksum errors: {checksum_errors} (Error Rate: {error_rate:.2f}%)")
        print(f"Unique valid packets recovered: {len(valid_packets)}/{total_packets}")
            
        # 4. RS Decode (Erasure Decoding)
        if not received_packets:
            print("Warning: No valid packets found. (Check SNR or Sync)")
            return b''
            
        # Reconstruct Block Structure
        # We need to find k_total from total_packets (N_total)
        n_total = total_packets
        k_total_est = 0
        
        # Search for k_total that matches n_total using a robust binary search
        low = 0
        high = n_total  # k_total cannot be larger than n_total
        k_total_est = -1
        blocks = []

        while low <= high:
            k_cand = (low + high) // 2
            if k_cand == 0:
                low = 1
                continue
            
            blks = self._calculate_block_structure(k_cand)
            calc_n = sum(k + p for k, p in blks)
            
            if calc_n == n_total:
                k_total_est = k_cand
                blocks = blks
                break
            elif calc_n < n_total:
                low = k_cand + 1
            else:
                high = k_cand - 1

        if k_total_est == -1:
            print("Warning: Could not determine exact block structure from total packets. Assuming single block (might fail).")
            blocks = [(n_total, 0)]  # Fallback
            k_total_est = n_total

        # Reconstruct parity mapping (Global Seq -> (Block, P_idx))
        parity_seq_map = {} 
        current_seq = k_total_est
        max_p = max(p for k,p in blocks) if blocks else 0
        for p_idx in range(max_p):
            for b_idx in range(len(blocks)):
                if p_idx < blocks[b_idx][1]:
                    parity_seq_map[current_seq] = (b_idx, p_idx)
                    current_seq += 1
        
        print(f"Decoding {len(blocks)} blocks...")
        start_time = time.time()
        
        # We will reconstruct data into a list of bytearrays, then join
        reconstructed_data_packets = [bytearray(self.packet_size) for _ in range(k_total_est)]

        for b_idx, (k_curr, n_parity_curr) in enumerate(blocks):
            n_curr = k_curr + n_parity_curr
            
            # Extract packets for this block
            block_packets = {}
            
            # 1. Data Packets (Round Robin)
            # Block b owns data packets: b, b+M, b+2M...
            num_blocks = len(blocks)
            for j in range(k_curr):
                seq = b_idx + j * num_blocks
                if seq in received_packets:
                    block_packets[j] = received_packets[seq]
            
            # 2. Parity Packets
            # We scan the parity_seq_map for this block.
            for seq, (blk, p_idx) in parity_seq_map.items():
                if blk == b_idx:
                    if seq in received_packets:
                        block_packets[k_curr + p_idx] = received_packets[seq]
            
            # RS Decode Block
            reconstructed_payloads = [bytearray(self.packet_size) for _ in range(k_curr)]
            
            rs = RSCodec(n_parity_curr) if n_parity_curr > 0 else None
            
            for i in range(self.packet_size):
                # Construct codeword with erasures
                codeword = bytearray(n_curr)
                erasures = []
                for j in range(n_curr):
                    if j in block_packets:
                        codeword[j] = block_packets[j][i]
                    else:
                        codeword[j] = 0
                        erasures.append(j)
                
                if rs:
                    try:
                        decoded_col = rs.decode(codeword, erase_pos=erasures)[0]
                        col_data = decoded_col
                    except Exception:
                        col_data = codeword[:k_curr]
                else:
                    col_data = codeword[:k_curr]
                
                # Place column data back into payloads
                for j in range(k_curr):
                    reconstructed_payloads[j][i] = col_data[j]
            
            # Place reconstructed payloads into global list
            for j, payload in enumerate(reconstructed_payloads):
                seq = b_idx + j * num_blocks
                if seq < k_total_est:
                    reconstructed_data_packets[seq] = payload
            
            if b_idx % 10 == 0:
                print_progress(b_idx + 1, len(blocks), start_time, prefix='RS Decode')
        
        print_progress(len(blocks), len(blocks), start_time, prefix='RS Decode')

        full_data = b''.join(reconstructed_data_packets)
        
        # Extract length prefix
        if len(full_data) >= 4:
            data_len = struct.unpack('>I', full_data[:4])[0]
            if data_len <= len(full_data) - 4:
                return full_data[4 : 4 + data_len]
        
        return full_data
