import os
import shutil
import subprocess
import sys
import patch

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


def dos2unix(filename):
    """Converts a file from DOS to Unix line endings in place."""
    with open(filename, 'rb') as f:
        content = f.read()
    
    # Don't write if the content is already in unix format
    if b'\r\n' not in content:
        return

    content = content.replace(b'\r\n', b'\n')
    with open(filename, 'wb') as f:
        f.write(content)

def main():
    """Main function to build and run ggwave."""
    rev = "3b877d07b102d8242a3fa9f333bddde464848f1b"

    if not os.path.exists("ggwave-to-file.exe"):
        if not os.path.isdir("ggwave"):
            run_command(["git", "clone", "https://github.com/ggerganov/ggwave"])

        ggwave_dir = "ggwave"
        
        run_command(["git", "checkout", "-f", rev], cwd=ggwave_dir)
        
        dos2unix(os.path.join(ggwave_dir, "CMakeLists.txt"))
        dos2unix(os.path.join(ggwave_dir, "examples", "CMakeLists.txt"))

        patch_file = f"patch_{rev}.diff"
        pset = patch.fromfile(patch_file)
        assert pset
        if not pset.apply(root=ggwave_dir):
            print(f"Error applying patch: {patch_file}", file=sys.stderr)
            sys.exit(1)

        build_dir = os.path.join(ggwave_dir, "b")
        if os.path.isdir(build_dir):
            shutil.rmtree(build_dir)

        run_command(["cmake", "-S", ".", "-B", "b"], cwd=ggwave_dir)
        run_command(["cmake", "--build", "b", "--config", "Release"], cwd=ggwave_dir)
        run_command(["ctest", ".", "-C", "Release"], cwd=build_dir)

        for exe_file in os.listdir(os.path.join(build_dir, "bin", "Release")):
            if exe_file.endswith(".exe"):
                shutil.copy(os.path.join(build_dir, "bin", "Release", exe_file), ".")

    for f in ["audio.wav", "audio.mp3"]:
        if os.path.exists(f):
            os.remove(f)
            
    p = subprocess.run(
        ["./ggwave-to-file", "-p0", "-v90"],
        input="This is a test message!",
        text=True,
        check=True,
    )

    run_command(["ffmpeg", "-i", "audio.wav", "-c:a", "libmp3lame", "-b:a", "320k", "-ac", "1", "-ar", "48000", "audio.mp3"])

    if os.path.exists("audio.wav"):
        os.remove("audio.wav")

    run_command(["ffmpeg", "-i", "audio.mp3", "-ac", "1", "-ar", "48000", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "audio.wav", "-y"])
    run_command(["./ggwave-from-file", "audio.wav"])

if __name__ == "__main__":
    main()
