import argparse
import sys
import numpy as np
from scipy.io import wavfile
from scipy import signal

def to_mono(data):
    """Converts stereo data to mono by averaging channels if necessary."""
    if len(data.shape) > 1:
        return np.mean(data, axis=1)
    return data

def load_wav(filename):
    """Loads a WAV file and normalizes it to float32 [-1, 1]."""
    fs, data = wavfile.read(filename)
    if data.dtype == np.float32 or data.dtype == np.float64:
        return fs, data.astype(np.float32)
    
    # Integer types
    try:
        max_val = np.iinfo(data.dtype).max
    except ValueError:
        # Fallback for unexpected types
        max_val = np.max(np.abs(data))
        if max_val == 0: max_val = 1.0
        
    return fs, data.astype(np.float32) / max_val

def align_signals(original, recorded):
    """
    Finds the lag between signals using cross-correlation and shifts the 
    recorded signal to align with the original.
    """
    # Use correlation to find the offset
    correlation = signal.correlate(recorded, original, mode='full')
    lags = signal.correlation_lags(recorded.size, original.size, mode='full')
    lag = lags[np.argmax(correlation)]

    print(f"Detected Lag: {lag} samples")

    # Shift the recorded signal
    if lag > 0:
        # Recording is delayed (normal case)
        aligned_recorded = recorded[lag:]
        # Trim original to match new length
        aligned_original = original[:len(aligned_recorded)]
    elif lag < 0:
        # Recording started before original (rare)
        aligned_original = original[abs(lag):]
        aligned_recorded = recorded[:len(aligned_original)]
    else:
        aligned_original = original
        aligned_recorded = recorded

    # Ensure lengths match perfectly
    min_len = min(len(aligned_original), len(aligned_recorded))
    return aligned_original[:min_len], aligned_recorded[:min_len]

def compute_impulse_response(original, recorded, regularization=1e-4):
    """
    Computes the Impulse Response (IR) using spectral division.
    H(f) = Y(f) / X(f)
    Regularization is added to avoid division by zero frequencies.
    """
    # FFT (Fast Fourier Transform)
    fft_in = np.fft.fft(original)
    fft_out = np.fft.fft(recorded)

    # Deconvolution with regularization (Wiener Deconvolution style)
    # H = (Y * conj(X)) / (|X|^2 + epsilon)
    numerator = fft_out * np.conj(fft_in)
    denominator = (np.abs(fft_in) ** 2) + regularization
    
    ir_freq = numerator / denominator
    
    # Inverse FFT to get back to time domain
    ir_time = np.fft.ifft(ir_freq)
    
    # Take real part (imaginary part should be near zero)
    return np.real(ir_time)

def generate_ir(args):
    print(f"Loading original: {args.original}")
    fs_orig, sig_orig = load_wav(args.original)
    print(f"Loading recorded: {args.recorded}")
    fs_rec, sig_rec = load_wav(args.recorded)

    if fs_orig != fs_rec:
        raise ValueError("Sample rates do not match! Please resample both files to the same rate (e.g., 44.1kHz).")

    # Convert to mono for IR extraction (Stereo IRs require processing L and R separately)
    sig_orig = to_mono(sig_orig)
    sig_rec = to_mono(sig_rec)

    print("Aligning signals...")
    orig_aligned, rec_aligned = align_signals(sig_orig, sig_rec)

    print("Computing Impulse Response...")
    impulse_response = compute_impulse_response(orig_aligned, rec_aligned)

    # Shift IR so the peak is in the middle (optional, but good for convolution plugins)
    peak_idx = np.argmax(np.abs(impulse_response))
    if peak_idx > len(impulse_response) // 2:
        impulse_response = np.roll(impulse_response, len(impulse_response) // 2)

    # Normalize IR Volume
    max_amp = np.max(np.abs(impulse_response))
    if max_amp > 0:
        impulse_response = impulse_response / max_amp

    # Save to WAV
    wavfile.write(args.output, fs_orig, impulse_response.astype(np.float32))
    print(f"Success! Model saved to: {args.output}")

def apply_ir(args):
    print(f"Loading input: {args.input}")
    fs_in, sig_in = load_wav(args.input)
    print(f"Loading IR: {args.impulse_response}")
    fs_ir, sig_ir = load_wav(args.impulse_response)

    if fs_in != fs_ir:
        print(f"Warning: Sample rates differ ({fs_in} vs {fs_ir}). Resampling IR to match input.")
        num_samples = int(len(sig_ir) * fs_in / fs_ir)
        sig_ir = signal.resample(sig_ir, num_samples)
        fs_ir = fs_in

    sig_ir = to_mono(sig_ir)

    print("Convolving...")
    if len(sig_in.shape) > 1:
        # Stereo input
        channels = []
        for i in range(sig_in.shape[1]):
            channels.append(signal.convolve(sig_in[:, i], sig_ir, mode='full'))
        output_signal = np.stack(channels, axis=1)
    else:
        output_signal = signal.convolve(sig_in, sig_ir, mode='full')

    # Normalize
    max_val = np.max(np.abs(output_signal))
    if max_val > 0:
        output_signal = output_signal / max_val

    print(f"Saving output to: {args.output}")
    wavfile.write(args.output, fs_in, output_signal.astype(np.float32))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Impulse Response Tool")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate Impulse Response from original and recorded files')
    gen_parser.add_argument('-o', '--original', required=True, help='Original audio file')
    gen_parser.add_argument('-r', '--recorded', required=True, help='Recorded audio file')
    gen_parser.add_argument('-out', '--output', required=True, help='Output Impulse Response WAV file')

    # Apply command
    apply_parser = subparsers.add_parser('apply', help='Apply Impulse Response to an audio file')
    apply_parser.add_argument('-i', '--input', required=True, help='Input audio file')
    apply_parser.add_argument('-ir', '--impulse-response', required=True, help='Impulse Response WAV file')
    apply_parser.add_argument('-out', '--output', required=True, help='Output convolved audio file')

    args = parser.parse_args()

    if args.command == 'generate':
        generate_ir(args)
    elif args.command == 'apply':
        apply_ir(args)