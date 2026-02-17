import unittest
import numpy as np
from ofdm.ofdm_protocol import OFDMProtocol

class TestOFDMProtocol(unittest.TestCase):
    def setUp(self):
        """Set up a default OFDMProtocol instance for tests."""
        self.ofdm = OFDMProtocol(n_fft=64, n_cp=16, pilot_interval=8)

    def test_init(self):
        """Test the initialization of the OFDMProtocol class."""
        self.assertEqual(self.ofdm.n_fft, 64)
        self.assertEqual(self.ofdm.n_cp, 16)
        self.assertEqual(self.ofdm.pilot_interval, 8)

        # Check subcarrier allocation
        self.assertTrue(np.array_equal(self.ofdm.sc_null, np.array([0, 32])))
        
        # Test there are no overlaps in subcarrier allocations
        sc_all = set(self.ofdm.sc_all)
        sc_data = set(self.ofdm.sc_data)
        sc_pilot = set(self.ofdm.sc_pilot)
        sc_null = set(self.ofdm.sc_null)

        self.assertEqual(len(sc_data.intersection(sc_pilot)), 0)
        self.assertEqual(len(sc_data.intersection(sc_null)), 0)
        self.assertEqual(len(sc_pilot.intersection(sc_null)), 0)
        
        # Test all subcarriers are accounted for
        self.assertEqual(len(sc_data) + len(sc_pilot) + len(sc_null), self.ofdm.n_fft)
        
        # Test preamble generation
        self.assertEqual(len(self.ofdm.preamble_t), self.ofdm.n_fft)
        self.assertTrue(np.iscomplexobj(self.ofdm.preamble_t))

    def test_bpsk_modulation(self):
        """Test BPSK modulation and demodulation."""
        bits = np.array([0, 1, 0, 1, 1, 0])
        symbols = self.ofdm.bpsk_mod(bits)
        expected_symbols = np.array([-1, 1, -1, 1, 1, -1])
        self.assertTrue(np.allclose(symbols, expected_symbols))
        
        demod_bits = self.ofdm.bpsk_demod(symbols)
        self.assertTrue(np.array_equal(bits, demod_bits))

    def test_qpsk_modulation(self):
        """Test QPSK modulation and demodulation."""
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        symbols = self.ofdm.qpsk_mod(bits)
        # 00->-1-1j, 01->-1+1j, 10->1-1j, 11->1+1j
        expected_symbols = np.array([-1-1j, -1+1j, 1-1j, 1+1j])
        self.assertTrue(np.allclose(symbols, expected_symbols))
        
        demod_bits = self.ofdm.qpsk_demod(symbols)
        self.assertTrue(np.array_equal(bits, demod_bits))

    def test_ask_modulation(self):
        """Test ASK modulation and demodulation."""
        bits = np.array([0, 1, 0, 1, 1, 0])
        symbols = self.ofdm.ask_mod(bits)
        expected_symbols = np.array([0+0j, 1+0j, 0+0j, 1+0j, 1+0j, 0+0j])
        self.assertTrue(np.allclose(symbols, expected_symbols))
        
        demod_bits = self.ofdm.ask_demod(symbols)
        self.assertTrue(np.array_equal(bits, demod_bits))

    def test_transmit_receive_bpsk(self):
        """Test end-to-end transmit and receive with BPSK modulation."""
        # Payload length that is not a multiple of data subcarriers
        payload = np.random.randint(0, 2, self.ofdm.n_data * 3 + 17)
        tx_signal = self.ofdm.transmit(payload, modulation='bpsk')
        
        # Noiseless channel
        rx_bits = self.ofdm.receive(tx_signal, modulation='bpsk')
        
        # The received bits should match the padded payload
        bits_per_symbol = self.ofdm.n_data
        n_syms = int(np.ceil(len(payload) / bits_per_symbol))
        padded_len = n_syms * bits_per_symbol
        
        self.assertEqual(len(rx_bits), padded_len)
        self.assertTrue(np.array_equal(payload, rx_bits[:len(payload)]))
        # Check padding is zeros
        self.assertTrue(np.all(rx_bits[len(payload):] == 0))

    def test_transmit_receive_qpsk(self):
        """Test end-to-end transmit and receive with QPSK modulation."""
        # Payload length that is a multiple of data subcarriers * 2
        payload = np.random.randint(0, 2, self.ofdm.n_data * 2 * 4)
        tx_signal = self.ofdm.transmit(payload, modulation='qpsk')
        
        # Noiseless channel
        rx_bits = self.ofdm.receive(tx_signal, modulation='qpsk')
        
        self.assertEqual(len(payload), len(rx_bits))
        self.assertTrue(np.array_equal(payload, rx_bits))

    def test_transmit_receive_ask(self):
        """Test end-to-end transmit and receive with ASK modulation."""
        payload = np.random.randint(0, 2, self.ofdm.n_data * 2)
        tx_signal = self.ofdm.transmit(payload, modulation='ask')
        
        rx_bits = self.ofdm.receive(tx_signal, modulation='ask')
        
        self.assertEqual(len(payload), len(rx_bits))
        self.assertTrue(np.array_equal(payload, rx_bits))

    def test_transmit_receive_with_noise(self):
        """Test end-to-end transmit and receive with some channel noise."""
        payload = np.random.randint(0, 2, self.ofdm.n_data * 2)
        tx_signal = self.ofdm.transmit(payload, modulation='qpsk')
        
        # Add a small amount of Gaussian noise
        noise_power = 0.01
        noise = np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal))
        noise *= np.sqrt(noise_power / 2)
        rx_signal = tx_signal + noise
        
        rx_bits = self.ofdm.receive(rx_signal, modulation='qpsk')
        
        self.assertEqual(len(payload), len(rx_bits))
        # With low noise, we expect no bit errors
        self.assertTrue(np.array_equal(payload, rx_bits))

    def test_multiple_transmissions(self):
        """Test receiving multiple concatenated transmissions."""
        payload1 = np.random.randint(0, 2, 100)
        payload2 = np.random.randint(0, 2, 150)
        
        tx_signal1 = self.ofdm.transmit(payload1, modulation='qpsk')
        tx_signal2 = self.ofdm.transmit(payload2, modulation='qpsk')
        
        full_signal = np.concatenate([tx_signal1, tx_signal2])
        
        rx_bits = self.ofdm.receive(full_signal, modulation='qpsk')
        
        # Calculate expected padded lengths for comparison
        bits_per_symbol = self.ofdm.n_data * 2
        n_syms1 = int(np.ceil(len(payload1) / bits_per_symbol))
        padded_len1 = n_syms1 * bits_per_symbol
        
        n_syms2 = int(np.ceil(len(payload2) / bits_per_symbol))
        padded_len2 = n_syms2 * bits_per_symbol
        
        total_padded_len = padded_len1 + padded_len2
        self.assertEqual(len(rx_bits), total_padded_len)

        # Check payload 1
        self.assertTrue(np.array_equal(payload1, rx_bits[:len(payload1)]))
        # Check payload 2
        p2_start_index = padded_len1
        self.assertTrue(np.array_equal(payload2, rx_bits[p2_start_index : p2_start_index+len(payload2)]))

    def test_large_payload_single_transmission(self):
        """Test a 1 KB payload to ensure it is treated as a single transmission."""
        # 1 KB = 1024 bytes = 8192 bits
        payload = np.random.randint(0, 2, 1024 * 8)
        tx_signal = self.ofdm.transmit(payload, modulation='qpsk')
        
        rx_bits = self.ofdm.receive(tx_signal, modulation='qpsk')
        
        # Calculate expected padded length for a single transmission
        bits_per_symbol = self.ofdm.n_data * 2  # QPSK
        n_syms = int(np.ceil(len(payload) / bits_per_symbol))
        expected_len = n_syms * bits_per_symbol
        
        # Verify that exactly one transmission was identified (length match)
        self.assertEqual(len(rx_bits), expected_len)
        self.assertTrue(np.array_equal(payload, rx_bits[:len(payload)]))

    def test_empty_payload(self):
        """Test with an empty payload."""
        payload = np.array([], dtype=int)
        tx_signal = self.ofdm.transmit(payload, modulation='qpsk')
        
        # The signal should only contain the preamble
        # Signal contains 3 repetitions of (Preamble + Header)
        symbol_len = self.ofdm.n_fft + self.ofdm.n_cp
        expected_len = 3 * (symbol_len + symbol_len)
        self.assertEqual(len(tx_signal), expected_len)

        rx_bits = self.ofdm.receive(tx_signal, modulation='qpsk')
        self.assertEqual(len(rx_bits), 0)

    def test_corrupted_preambles(self):
        """Test robustness against corruption of 2 out of 3 preamble/header repetitions."""
        payload = np.random.randint(0, 2, 100)
        tx_signal = self.ofdm.transmit(payload, modulation='qpsk')
        
        symbol_len = self.ofdm.n_fft + self.ofdm.n_cp
        rep_len = 2 * symbol_len # Preamble + Header
        
        # Case 1: Corrupt first two (Rep 0 and Rep 1)
        rx_signal_1 = tx_signal.copy()
        rx_signal_1[0 : 2 * rep_len] = 0
        
        rx_bits_1 = self.ofdm.receive(rx_signal_1, modulation='qpsk')
        self.assertGreater(len(rx_bits_1), 0, "Failed to decode with first two preambles corrupted")
        self.assertTrue(np.array_equal(payload, rx_bits_1[:len(payload)]))

        # Case 2: Corrupt first and last (Rep 0 and Rep 2)
        rx_signal_2 = tx_signal.copy()
        rx_signal_2[0 : rep_len] = 0 # Rep 0
        rx_signal_2[2 * rep_len : 3 * rep_len] = 0 # Rep 2
        
        rx_bits_2 = self.ofdm.receive(rx_signal_2, modulation='qpsk')
        self.assertGreater(len(rx_bits_2), 0, "Failed to decode with first and last preambles corrupted")
        self.assertTrue(np.array_equal(payload, rx_bits_2[:len(payload)]))
        
        # Case 3: Corrupt last two (Rep 1 and Rep 2)
        rx_signal_3 = tx_signal.copy()
        rx_signal_3[rep_len : 3 * rep_len] = 0 # Rep 1 and Rep 2
        
        rx_bits_3 = self.ofdm.receive(rx_signal_3, modulation='qpsk')
        self.assertGreater(len(rx_bits_3), 0, "Failed to decode with last two preambles corrupted")
        self.assertTrue(np.array_equal(payload, rx_bits_3[:len(payload)]))

    def test_missing_preambles_cut(self):
        """Test robustness when preamble repetitions are missing (cut out) from the signal."""
        payload = np.random.randint(0, 2, 100)
        tx_signal = self.ofdm.transmit(payload, modulation='qpsk')
        
        symbol_len = self.ofdm.n_fft + self.ofdm.n_cp
        rep_len = 2 * symbol_len # Preamble + Header
        
        # Case 1: Cut first repetition (Rep 0) entirely
        # Original: [Rep0][Rep1][Rep2][Data]
        # New:      [Rep1][Rep2][Data]
        rx_signal_1 = tx_signal[rep_len:]
        
        rx_bits_1 = self.ofdm.receive(rx_signal_1, modulation='qpsk')
        self.assertGreater(len(rx_bits_1), 0, "Failed to decode with first repetition cut")
        self.assertTrue(np.array_equal(payload, rx_bits_1[:len(payload)]))

        # Case 2: Cut first two repetitions (Rep 0 and Rep 1)
        # Original: [Rep0][Rep1][Rep2][Data]
        # New:      [Rep2][Data]
        rx_signal_2 = tx_signal[2 * rep_len:]
        
        rx_bits_2 = self.ofdm.receive(rx_signal_2, modulation='qpsk')
        self.assertGreater(len(rx_bits_2), 0, "Failed to decode with first two repetitions cut")
        self.assertTrue(np.array_equal(payload, rx_bits_2[:len(payload)]))

if __name__ == '__main__':
    unittest.main()
