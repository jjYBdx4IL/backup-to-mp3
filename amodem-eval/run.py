#!/usr/bin/env python3
import argparse
import filecmp
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.resolve()
VENV_DIR = BASE_DIR / "venv"
TEMP_DIR = tempfile.TemporaryDirectory()

VALID_BITRATES = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 42, 48, 54, 60, 64, 72, 80]

def get_sample_rate(bitrate):
    if bitrate < 12:
        return 8000
    elif bitrate < 28:
        return 16000
    else:
        return 32000

def setup_environment():
    """Ensures venv exists and returns the path to the python executable."""
    if not VENV_DIR.exists():
        print("Virtual environment not found. Running setup_venv.py...")
        setup_script = BASE_DIR / "setup_venv.py"
        if not setup_script.exists():
            raise FileNotFoundError(f"{setup_script} not found.")
        subprocess.check_call([sys.executable, str(setup_script)])

    if platform.system() == "Windows":
        venv_python = VENV_DIR / "Scripts" / "python.exe"
    else:
        venv_python = VENV_DIR / "bin" / "python"

    if not venv_python.exists():
        raise FileNotFoundError(f"Python executable not found at {venv_python}")
    
    return venv_python

def run_encode(venv_python, input_file, output_file, bitrate, sample_rate, debug=False):
    print(f"--- Encoding {input_file} to {output_file} ---")
    temp_dir = Path(TEMP_DIR.name)
    pcm_file = temp_dir / "encode_temp.pcm"
    
    env = os.environ.copy()
    env["BITRATE"] = str(bitrate)

    # 1. amodem send
    cmd_send = [str(venv_python), "-m", "amodem", "send", "--silence", "1", "-i", str(input_file), "-o", str(pcm_file)]
    print(f"Running: {' '.join(cmd_send)} (BITRATE={bitrate})")
    
    process = subprocess.Popen(cmd_send, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    stdout, _ = process.communicate()
    print(stdout)
    
    if process.returncode != 0:
        raise RuntimeError("amodem send failed.")

    print(f"Using sample rate: {sample_rate} Hz")

    # 2. ffmpeg PCM -> MP3
    cmd_ffmpeg = [
        "ffmpeg", "-n",
        "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-channel_layout", "mono",
        "-i", str(pcm_file),
        "-c:a", "libmp3lame", "-b:a", "320k", "-map_metadata", "-1",
        "-ac", "1", "-ar", "32000", #str(sample_rate),
        str(output_file)
    ]
    print(f"Running: {' '.join(cmd_ffmpeg)}")
    if debug:
        subprocess.check_call(cmd_ffmpeg)
    else:
        try:
            subprocess.check_output(cmd_ffmpeg, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf-8', errors='replace'))
            raise

    # 3. Verification (Test Decode)
    print("\n--- Verifying Encode ---")
    verify_output = temp_dir / "verify_output.dat"
    if verify_output.exists():
        verify_output.unlink()

    cmd_decode = [
        sys.executable, str(BASE_DIR / "run.py"), "decode",
        "--bitrate", str(bitrate),
        "-i", str(output_file), "-o", str(verify_output)
    ]
    if debug:
        cmd_decode.append("--debug")
    print(f"Running: {' '.join(cmd_decode)}")
    subprocess.check_call(cmd_decode)

    # Compare
    if filecmp.cmp(input_file, verify_output, shallow=False):
        print("Verification SUCCESS: Decoded file matches input.")
    else:
        if output_file.exists():
            output_file.unlink()
        raise RuntimeError("Verification FAILED: Decoded file differs from input.")

def run_decode(venv_python, input_file, output_file, bitrate, sample_rate, debug=False):
    print(f"--- Decoding {input_file} to {output_file} ---")
    temp_dir = Path(TEMP_DIR.name)
    pcm_file = temp_dir / "decode_temp.pcm"

    env = os.environ.copy()
    env["BITRATE"] = str(bitrate)

    # 1. ffmpeg MP3 -> PCM
    cmd_ffmpeg = [
        "ffmpeg",
        "-y",
        "-i", str(input_file),
        "-ac", "1", "-ar", str(sample_rate),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-f", "s16le",
        str(pcm_file)
    ]
    print(f"Running: {' '.join(cmd_ffmpeg)}")
    if debug:
        subprocess.check_call(cmd_ffmpeg)
    else:
        try:
            subprocess.check_output(cmd_ffmpeg, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf-8', errors='replace'))
            raise

    # 2. amodem recv
    cmd_recv = [str(venv_python), "-m", "amodem", "recv", "-i", str(pcm_file), "-o", str(output_file), "--ignore-checksum-errors"]
    print(f"Running: {' '.join(cmd_recv)} (BITRATE={bitrate})")

    process = subprocess.Popen(cmd_recv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    stdout, _ = process.communicate()
    print(stdout)

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd_recv)

    lines = stdout.strip().splitlines()
    if lines and not re.search(r"Received [\d\.]+ kB @ [\d\.]+ seconds = [\d\.]+ kB/s", lines[-1]):
        raise Exception("Last line of amodem recv output does not match expected format.")

def main():
    parser = argparse.ArgumentParser(description="Frontend for amodem with ffmpeg conversion.")
    parser.add_argument("action", choices=["encode", "decode"], help="Action to perform")
    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    
    parser.add_argument("-b", "--bitrate", type=int, 
                        choices=[0] + VALID_BITRATES, 
                        default=0,
                        help="Bitrate in kbits/s")
    parser.add_argument("--debug", action="store_true", help="Show ffmpeg output")
    
    args = parser.parse_args()

    if args.bitrate is None:
        parser.error("the following arguments are required: -b/--bitrate")

    venv_python = setup_environment()
    
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist.")

    if output_path.exists():
        raise FileExistsError(f"Output file {output_path} already exists.")

    if args.action == "encode":
        if args.bitrate == 0:
            parser.error("Bitrate 0 is not allowed for encoding.")
        sample_rate = get_sample_rate(args.bitrate)
        run_encode(venv_python, input_path, output_path, args.bitrate, sample_rate, debug=args.debug)
    else:
        if args.bitrate == 0:
            print("Bitrate set to 0. Attempting to auto-detect bitrate...")
            success = False
            for b in VALID_BITRATES:
                sr = get_sample_rate(b)
                print(f"\nTrying bitrate: {b} (Sample Rate: {sr})")
                try:
                    run_decode(venv_python, input_path, output_path, b, sr, debug=args.debug)
                    print(f"Success with bitrate: {b}")
                    success = True
                    break
                except:
                    print(f"Failed with bitrate: {b}")
            if not success:
                raise Exception("Auto-detection failed: Could not decode with any bitrate.")
        else:
            sample_rate = get_sample_rate(args.bitrate)
            run_decode(venv_python, input_path, output_path, args.bitrate, sample_rate, debug=args.debug)

if __name__ == "__main__":
    main()