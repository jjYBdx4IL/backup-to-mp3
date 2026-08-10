#!/usr/bin/env python3
import os
import sys
import subprocess
import base64
import re
import hashlib
import argparse
import tempfile
import json
from collections import Counter

def check_overwrite(path, force):
    if os.path.exists(path) and not force:
        print(f"Error: output file '{path}' already exists. Use -f/--force to overwrite.", file=sys.stderr)
        sys.exit(1)

def get_direwolf_version():
    result = subprocess.run(["direwolf", "--version"], capture_output=True, text=True)
    first_line = result.stdout.splitlines()[0].strip() if result.stdout else ""
    match = re.match(r'^Dire Wolf version (\S+)$', first_line)
    if not match:
        raise RuntimeError(f"Unexpected 'direwolf --version' output: {first_line!r}")
    return match.group(1)

def encode_file(input_bin_path, output_wav_path, chunk_size=150, fx25_redundancy=64, repeat=3, force=False):
    check_overwrite(output_wav_path, force)

    file_name = os.path.basename(input_bin_path)
    file_size = os.path.getsize(input_bin_path)

    with open(input_bin_path, "rb") as f:
        binary_data = f.read()

    md5sum = hashlib.md5(binary_data).hexdigest()
    b64_encoded = base64.b64encode(binary_data).decode('ascii')
    chunks = [b64_encoded[i:i + chunk_size] for i in range(0, len(b64_encoded), chunk_size)]
    num_blocks = len(chunks)

    print(f"Encoding File: {file_name} ({file_size} bytes, {num_blocks} blocks, md5={md5sum})")
    print(f"Repeating transmission {repeat} time(s)")

    to_mp3 = output_wav_path.lower().endswith(".mp3")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_txt_path = os.path.join(tmpdir, "packets.txt")
        gen_wav_path = os.path.join(tmpdir, "output.wav") if to_mp3 else output_wav_path

        try:
            header_payload = f"H:{file_name}:{file_size}:{num_blocks}:{md5sum}"
            packet_lines = [f"S>D:{header_payload}"]
            for idx, chunk in enumerate(chunks):
                packet_lines.append(f"S>D:{idx}:{chunk}")

            with open(temp_txt_path, "w") as out:
                for _ in range(repeat):
                    for line in packet_lines:
                        out.write(line + "\n")

            baud_rate = 300
            gen_packets_params = f"-B {baud_rate} -X {fx25_redundancy}"
            cmd = ["gen_packets", "-B", str(baud_rate), "-o", gen_wav_path, "-X", str(fx25_redundancy), temp_txt_path]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Successfully generated WAV: {gen_wav_path}")

            if to_mp3:
                direwolf_version = get_direwolf_version()
                id3_title = os.path.basename(output_wav_path)
                id3_comment = (f"name={file_name};size={file_size};blocks={num_blocks};md5={md5sum};"
                                f"gen_packets={gen_packets_params};direwolf={direwolf_version}")
                ffmpeg_cmd = ["ffmpeg", "-i", gen_wav_path, "-c:a", "libmp3lame",
                              "-b:a", "64k", "-ac", "1",
                              "-metadata", f"title={id3_title}",
                              "-metadata", f"comment={id3_comment}",
                              output_wav_path]
                print(f"Running: {' '.join(ffmpeg_cmd)}")
                subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
                print(f"Successfully generated MP3: {output_wav_path}")
                print(f"NOTE: embedded ID3 tags (title={id3_title}, comment={id3_comment}) into the MP3.")
        except subprocess.CalledProcessError as e:
            print(f"Error running {e.cmd[0]}: {e}", file=sys.stderr)
            if e.stderr:
                print(e.stderr, file=sys.stderr)
            sys.exit(1)

def read_mp3_id3_metadata(mp3_path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format_tags=title,comment", "-of", "json", mp3_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        tags = json.loads(result.stdout).get("format", {}).get("tags", {})
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return {}

    info = {}
    comment_keys = ("name", "size", "blocks", "md5", "gen_packets", "direwolf")
    for field in tags.get("comment", "").split(";"):
        key, _, value = field.partition("=")
        if key in comment_keys and value:
            info[key] = value
    return info

def decode_wav(input_wav_path, output_bin_path=None, trace=False, force=False):
    print(f"Decoding WAV file: {input_wav_path} using atest...")

    if output_bin_path:
        check_overwrite(output_bin_path, force)

    id3_info = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        atest_input_path = input_wav_path
        if input_wav_path.lower().endswith(".mp3"):
            id3_info = read_mp3_id3_metadata(input_wav_path)
            if id3_info:
                print(f"NOTE: found ID3 metadata in MP3 (informational only, not verified against audio): "
                      f"name={id3_info.get('name')}, size={id3_info.get('size')}, "
                      f"blocks={id3_info.get('blocks')}, md5={id3_info.get('md5')}, "
                      f"gen_packets={id3_info.get('gen_packets')}, direwolf={id3_info.get('direwolf')}")
            else:
                print("NOTE: no usable ID3 metadata found in MP3; decoding relies solely on the "
                      "audio-encoded header.")

            atest_input_path = os.path.join(tmpdir, "decoded.wav")
            ffmpeg_cmd = ["ffmpeg", "-i", input_wav_path, "-map_metadata", "-1",
                          "-fflags", "+bitexact", "-flags:a", "+bitexact",
                          "-c:a", "pcm_s16le", atest_input_path]
            print(f"Running: {' '.join(ffmpeg_cmd)}")
            try:
                subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running ffmpeg: {e}", file=sys.stderr)
                if e.stderr:
                    print(e.stderr, file=sys.stderr)
                sys.exit(1)

        cmd = ["atest", "-B", "300", atest_input_path]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output_lines = result.stdout.splitlines()
        except subprocess.CalledProcessError as e:
            print(f"Error running atest: {e}", file=sys.stderr)
            if e.stderr:
                print(e.stderr, file=sys.stderr)
            sys.exit(1)
    
    file_name = None
    file_size = None
    expected_blocks = None
    expected_md5 = None
    blocks = {}
    block_counts = {}

    ansi_re = re.compile(r'\x1b\[[0-9;]*m')
    header_re = re.compile(r'^\[\d+\]\s+S>D:H:(?P<name>[^:<]+):(?P<size>\d+):(?P<blocks>\d+)(?::(?P<md5>[0-9a-fA-F]+))?')
    data_re = re.compile(r'^\[\d+\]\s+S>D:(\d+):(.*)')

    for raw_line in output_lines:
        line = ansi_re.sub('', raw_line)
        if trace:
            print(f"ATEST Output: {line.strip()}")
        header_match = header_re.match(line)
        if header_match:
            file_name = header_match.group('name').strip()
            file_size = int(header_match.group('size'))
            expected_blocks = int(header_match.group('blocks'))
            expected_md5 = header_match.group('md5')
            header_msg = f"Found Header -> File: {file_name}, Size: {file_size} bytes, Blocks: {expected_blocks}"
            if expected_md5:
                header_msg += f", MD5: {expected_md5}"
            print(header_msg)

        elif data_re.match(line):
            try:
                match = data_re.match(line)
                idx = int(match.group(1))
                chunk_data = match.group(2).split('<')[0].strip()
                if idx in blocks and blocks[idx] != chunk_data:
                    print(f"Warning: Conflicting duplicate data for block {idx}; "
                          f"keeping first-received copy.", file=sys.stderr)
                else:
                    blocks[idx] = chunk_data
                block_counts[idx] = block_counts.get(idx, 0) + 1
            except Exception as e:
                print(f"Warning: Failed to parse data block line: {line.strip()} ({e})")

    if not expected_blocks:
        if id3_info.get("blocks"):
            print(f"WARNING: no valid header packet (H:) found in the decoded audio; falling back to "
                  f"block count from MP3 ID3 metadata (blocks={id3_info['blocks']}). This fallback is "
                  f"unverified until the MD5 check below completes.", file=sys.stderr)
            expected_blocks = int(id3_info["blocks"])
            file_name = file_name or id3_info.get("name")
            expected_md5 = expected_md5 or id3_info.get("md5")
        else:
            print("Error: No valid header packet (H:) found in the decoded audio.", file=sys.stderr)
            sys.exit(1)

    print(f"Recovered {len(blocks)} of {expected_blocks} expected blocks.")
    
    b64_sorted_chunks = []
    for i in range(expected_blocks):
        if i in blocks:
            b64_sorted_chunks.append(blocks[i])
        else:
            print(f"Warning: Missing block index {i}", file=sys.stderr)
            b64_sorted_chunks.append("")
            
    full_b64 = "".join(b64_sorted_chunks)
    
    try:
        binary_data = base64.b64decode(full_b64)
    except Exception as e:
        print(f"Error decoding base64 data: {e}", file=sys.stderr)
        sys.exit(1)
        
    out_path = output_bin_path if output_bin_path else file_name or "recovered_file.bin"
    check_overwrite(out_path, force)

    with open(out_path, "wb") as f:
        f.write(binary_data)

    print(f"Successfully reconstructed file: {out_path} ({len(binary_data)} bytes)")

    if expected_md5:
        actual_md5 = hashlib.md5(binary_data).hexdigest()
        if actual_md5 == expected_md5:
            print(f"MD5 verification passed: {actual_md5}")
        else:
            print(f"MD5 verification FAILED: expected {expected_md5}, got {actual_md5}", file=sys.stderr)
    else:
        print("Warning: No MD5 checksum found in header; skipping verification.", file=sys.stderr)

    print("\nBlock reception statistics:")
    reception_histogram = Counter(block_counts.get(i, 0) for i in range(expected_blocks))
    for count in sorted(reception_histogram):
        n = reception_histogram[count]
        label = "block" if n == 1 else "blocks"
        if count == 0:
            print(f"  received 0 times (missing): {n} {label}")
        else:
            print(f"  received {count} time(s): {n} {label}")

def main():
    parser = argparse.ArgumentParser(prog="direwolf_tool.py")
    subparsers = parser.add_subparsers(dest="mode")

    encode_parser = subparsers.add_parser("encode", help="Encode a binary file into a WAV of FX.25 packets")
    encode_parser.add_argument("input_bin_path")
    encode_parser.add_argument("output_wav_path")
    encode_parser.add_argument("-R", "--repeat", type=int, default=3,
                                help="Number of times to repeat the entire transmission (default: 3)")
    encode_parser.add_argument("-f", "--force", action="store_true",
                                help="Overwrite the output file if it already exists")

    decode_parser = subparsers.add_parser("decode", help="Decode a WAV of FX.25 packets back into a binary file")
    decode_parser.add_argument("input_wav_path")
    decode_parser.add_argument("output_bin_path", nargs="?", default=None)
    decode_parser.add_argument("--trace", action="store_true",
                                help="Print each raw atest output line during decoding")
    decode_parser.add_argument("-f", "--force", action="store_true",
                                help="Overwrite the output file if it already exists")

    args = parser.parse_args()

    if args.mode == "encode":
        if args.repeat < 1:
            parser.error("-R/--repeat must be at least 1")
        encode_file(args.input_bin_path, args.output_wav_path, repeat=args.repeat, force=args.force)
    elif args.mode == "decode":
        decode_wav(args.input_wav_path, args.output_bin_path, trace=args.trace, force=args.force)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
