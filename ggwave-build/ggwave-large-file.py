import sys
import argparse
import os
import math
import base64
import subprocess
import wave
import tempfile
import shutil

TMPDIR = tempfile.TemporaryDirectory()

CHUNK_SIZE = 90
SILENCE_DURATION = 0.5

def find_executable(name):
    if os.path.exists(name): return os.path.abspath(name)
    if os.path.exists("./" + name): return os.path.abspath("./" + name)
    return name

GGWAVE_TO_FILE = find_executable("ggwave-to-file.exe")
GGWAVE_FROM_FILE = find_executable("ggwave-from-file.exe")

def run_command(command, cwd=None):
    """Runs a command in the shell and checks for errors."""
    print(f"Executing: {' '.join(command)}")
    process = subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True)
    if process.returncode != 0:
        print(f"Error executing command: {' '.join(command)}", file=sys.stderr)
        print(process.stdout, file=sys.stdout)
        print(process.stderr, file=sys.stderr)
        sys.exit(1)
    print(process.stdout)
    
def decode_file(args):
    if os.path.exists(args.output_file) and not args.overwrite:
        print(f"Error: Output file '{args.output_file}' already exists. Use -y to overwrite.")
        sys.exit(1)

    # 1. Test encode max block size to determine duration
    print("Determining block duration...")
    # Max 140 chars as per constraint
    test_payload = "x" * 140
    test_wav = os.path.join(TMPDIR.name, "audio.wav")
    if os.path.exists(test_wav): os.remove(test_wav)
    
    try:
        cmd = [GGWAVE_TO_FILE, f"-p{args.protocol}", "-v90", "-s48000"] + (["-d"] if args.dss else [])
        if args.verbose:
            print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, input=test_payload, cwd=TMPDIR.name, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Error running {GGWAVE_TO_FILE}: {e}")
        sys.exit(1)
        
    block_duration = 0.0
    if os.path.exists(test_wav):
        with wave.open(test_wav, 'rb') as w:
            block_duration = w.getnframes() / w.getframerate()
        os.remove(test_wav)
    else:
        print("Failed to generate test audio block.")
        sys.exit(1)
        
    print(f"Block duration: {block_duration:.3f}s")
    
    # 2. Define overlap and window
    window_sec = block_duration * 2.0
    step_sec = block_duration * 0.5
    
    print(f"Using window: {window_sec:.3f}s, step: {step_sec:.3f}s")
    
    input_wav = args.input_wav
    if not input_wav.lower().endswith(".wav"):
        converted_wav = os.path.join(TMPDIR.name, "converted.wav")
        run_command(["ffmpeg", "-i", input_wav, "-ac", "1", "-ar", "48000", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", converted_wav, "-y"])
        input_wav = converted_wav

    # 3. Scan input file
    decoded_blocks = {}
    total_blocks = 0
    next_expected_index = 1
    
    with wave.open(input_wav, 'rb') as w:
        params = w.getparams()
        total_frames = w.getnframes()
        framerate = w.getframerate()
        duration = total_frames / framerate
        
        pos = 0.0
        while pos < duration:
            w.setpos(int(pos * framerate))
            frames = w.readframes(int(window_sec * framerate))
            
            temp_chunk = os.path.join(TMPDIR.name, "temp.wav")
            with wave.open(temp_chunk, 'wb') as tmp:
                tmp.setparams(params)
                tmp.writeframes(frames)
            
            cmd = [GGWAVE_FROM_FILE] +  (["-d"] if args.dss else []) + [temp_chunk]
            if args.verbose:
                print(f"Executing: {' '.join(cmd)}")
            res = subprocess.run(cmd, capture_output=True, text=True)
            output = res.stdout.strip()
            
            if output:
                for line in output.splitlines():
                    if "Decoded message with length" in line:
                        try:
                            start = line.find("'")
                            end = line.rfind("'")
                            if start != -1 and end != -1 and end > start:
                                content = line[start+1:end]
                                parts = content.strip().split(' ')
                                if len(parts) == 2 and '/' in parts[0]:
                                    idx_str, total_str = parts[0].split('/')
                                    idx = int(idx_str)
                                    current_total_blocks = int(total_str)
                                    
                                    if total_blocks == 0:
                                        total_blocks = current_total_blocks
                                    
                                    if idx == next_expected_index:
                                        decoded_blocks[idx] = parts[1]
                                        print(f"\rDecoded block {idx}/{total_blocks} at {pos:.1f}s", end='')
                                        next_expected_index += 1
                                    elif idx > next_expected_index:
                                        print(f"\nWarning: Out of order block. Expected {next_expected_index}, got {idx} at {pos:.1f}s")
                        except:
                            pass
            
            if total_blocks > 0 and next_expected_index > total_blocks:
                break
            
            pos += step_sec
            
    temp_chunk_path = os.path.join(TMPDIR.name, "temp.wav")
    if os.path.exists(temp_chunk_path): os.remove(temp_chunk_path)
    print("")
    
    if total_blocks == 0 or len(decoded_blocks) != total_blocks:
        print(f"\nError: Did not receive all blocks. Expected {total_blocks}, got {len(decoded_blocks)}.")
        sys.exit(1)
        
    # Reassemble
    print(f"Reassembling {len(decoded_blocks)} blocks...")
    final_data = bytearray()
    for i in sorted(decoded_blocks.keys()):
        try:
            final_data.extend(base64.b64decode(decoded_blocks[i]))
        except:
            print(f"Error decoding base64 for block {i}")
            
    with open(args.output_file, 'wb') as f:
        f.write(final_data)
    print(f"Decoded data written to {args.output_file}")

def encode_file(args):
    if os.path.exists(args.output_file) and not args.overwrite:
        print(f"Error: Output file '{args.output_file}' already exists. Use -y to overwrite.")
        sys.exit(1)

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    with open(args.input_file, 'rb') as f:
        data = f.read()

    total_chunks = math.ceil(len(data) / CHUNK_SIZE)
    if total_chunks == 0:
        print("Input file is empty.")
        return

    print(f"Encoding {len(data)} bytes into {total_chunks} chunks...")

    output_wav = None
    temp_wav_path = os.path.join(TMPDIR.name, "temp_output.wav")
    chunk_durations = []

    for i in range(total_chunks):
        chunk = data[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        b64_chunk = base64.b64encode(chunk).decode('utf-8')
        
        # Format: INDEX/TOTAL BASE64
        # We use 1-based indexing for the blocks
        message = f"{i+1}/{total_chunks} {b64_chunk}"
        
        # Run ggwave-to-file
        # We assume it writes to audio.wav in the current directory (which we set to TMPDIR)
        audio_wav = os.path.join(TMPDIR.name, "audio.wav")
        if os.path.exists(audio_wav):
            os.remove(audio_wav)
            
        try:
            cmd = [GGWAVE_TO_FILE, f"-p{args.protocol}", "-v90", "-s48000"] + (["-d"] if args.dss else [])
            if args.verbose:
                print(f"Executing: {' '.join(cmd)}")
            subprocess.run(
                cmd, input=message,
                cwd=TMPDIR.name,
                text=True,
                check=True,
                capture_output=True
            )
        except FileNotFoundError:
             print(f"Error: {GGWAVE_TO_FILE} not found.")
             sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Error running ggwave-to-file for chunk {i+1}: {e}")
            print(f"Stderr: {e.stderr}")
            sys.exit(1)

        if not os.path.exists(audio_wav):
            print(f"Error: audio.wav not generated for chunk {i+1}")
            sys.exit(1)

        with wave.open(audio_wav, 'rb') as w:
            chunk_durations.append(w.getnframes() / w.getframerate())
            if output_wav is None:
                output_wav = wave.open(temp_wav_path, 'wb')
                output_wav.setparams(w.getparams())
            
            frames = w.readframes(w.getnframes())
            output_wav.writeframes(frames)
            
            # Add silence between blocks to help separation
            silence_frames = int(w.getframerate() * SILENCE_DURATION)
            silence = b'\x00' * (silence_frames * w.getnchannels() * w.getsampwidth())
            output_wav.writeframes(silence)
            
        print(f"Processed chunk {i+1}/{total_chunks}", end='\r')

    print("") # Newline after progress

    if len(chunk_durations) > 1:
        stats = chunk_durations[:-1]
        print(f"Stats (excluding last): Min: {min(stats):.2f}s, Avg: {sum(stats)/len(stats):.2f}s, Max: {max(stats):.2f}s")
    elif len(chunk_durations) == 1:
        print("Only one chunk generated, cannot exclude last for stats.")

    if output_wav:
        output_wav.close()
        print(f"Done. Audio generated.")

        with wave.open(temp_wav_path, 'rb') as w:
            duration = w.getnframes() / w.getframerate()
            channels = w.getnchannels()
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = duration % 60
            print(f"WAV Duration: {hours}h {minutes}m {seconds:.2f}s")
            if duration > 0:
                print(f"Encoding Speed: {len(data) / duration:.2f} bytes/second")
            print(f"WAV Channels: {channels}")
        
        if args.output_file.lower().endswith(".mp3"):
            run_command(["ffmpeg", "-i", temp_wav_path, "-c:a", "libmp3lame", "-b:a", args.bitrate, "-ac", "1", "-ar", "48000", "-y", args.output_file])
            
            if os.path.exists(args.output_file):
                size = os.path.getsize(args.output_file)
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if size < 1024:
                        break
                    size /= 1024
                print(f"MP3 Size: {size:.2f} {unit}")
        else:
            if os.path.exists(args.output_file):
                os.remove(args.output_file)
            shutil.copyfile(temp_wav_path, args.output_file)
            print(f"Output written to {args.output_file}")

        print("Verifying...")
        verify_output = os.path.join(TMPDIR.name, "verify_output.dat")
        
        verify_cmd = [sys.executable, os.path.abspath(__file__), "decode", args.output_file, verify_output]
        verify_cmd.append(f"-p{args.protocol}")
        if args.dss:
            verify_cmd.append("--dss")
        if args.verbose:
            verify_cmd.append("-v")
            
        try:
            subprocess.run(verify_cmd, check=True)
            
            if os.path.exists(verify_output):
                with open(verify_output, 'rb') as f:
                    decoded_data = f.read()
                
                if decoded_data == data:
                    print("Verification successful: Decoded data matches input.")
                else:
                    # make sure to display some nasty traces if not ok
                    raise Exception("Verification failed: Decoded data does not match input.")
            else:
                raise Exception("Verification failed: Output file not generated.")
        except subprocess.CalledProcessError:
            raise Exception("Verification failed: Decode process returned error.")
    else:
        raise Exception("No audio generated.")
    # and otherwise a nice and concise OK
    print("OK")

def main():
    parser = argparse.ArgumentParser(description='Encode/Decode large files with ggwave.')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('-p', '--protocol', default='2', help='GGWave protocol (default: 2)')
    common_parser.add_argument('--dss', action='store_true', help='enable dss for ggwave encoder/decoder runs')
    common_parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose output')
    common_parser.add_argument('-y', '--overwrite', action='store_true', help='overwrite output files')

    # Encode parser
    enc_parser = subparsers.add_parser('encode', parents=[common_parser], help='Encode a file')
    enc_parser.add_argument('input_file', help='Input file to encode')
    enc_parser.add_argument('output_file', nargs='?', default='output.wav', help='Output WAV file')
    enc_parser.add_argument('-b', '--bitrate', default='64k', help='MP3 bitrate (default: 64k)')

    # Decode parser
    dec_parser = subparsers.add_parser('decode', parents=[common_parser], help='Decode a file')
    dec_parser.add_argument('input_wav', help='Input WAV file to decode')
    dec_parser.add_argument('output_file', nargs='?', default='output.dat', help='Output file')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'encode':
        encode_file(args)
    elif args.command == 'decode':
        decode_file(args)

if __name__ == "__main__":
    main()
