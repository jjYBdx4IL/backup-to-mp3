#!/usr/bin/env python3
import platform
import sys
import subprocess
import venv
from pathlib import Path
import patch

REV="v1.16.0"

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
    base_dir = Path(__file__).parent
    req_file = base_dir / "requirements.txt"
    venv_dir = base_dir / "venv"

    if not req_file.exists():
        print(f"Error: 'requirements.txt' not found in {base_dir}")
        sys.exit(1)

    print(f"Creating virtual environment in {venv_dir}...")
    # clear=True ensures we start fresh if the folder exists
    venv.create(venv_dir, with_pip=True, clear=True)

    if sys.platform == "win32":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    # Check if pip is installed, if not try to bootstrap it
    try:
        subprocess.check_call([str(venv_python), "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("pip not found. Bootstrapping with ensurepip...")
        subprocess.check_call([str(venv_python), "-m", "ensurepip", "--upgrade", "--default-pip"])

    print("Upgrading pip...")
    subprocess.check_call([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])

    subprocess.check_call("git clone https://github.com/romanz/amodem.git".split(" "))
    amodem_dir = base_dir / "amodem"
    subprocess.check_call(f"git checkout -f {REV}".split(" "), cwd=str(amodem_dir))

    # dos2unix(os.path.join(ggwave_dir, "CMakeLists.txt"))
    # dos2unix(os.path.join(ggwave_dir, "examples", "CMakeLists.txt"))

    patch_file = base_dir / f"patch_{REV.replace('.', '_')}.diff"
    pset = patch.fromfile(patch_file)
    assert pset
    if not pset.apply(root=amodem_dir):
        print(f"Error applying patch: {patch_file}", file=sys.stderr)
        sys.exit(1)

    run_command(f"{venv_python} -m pip install -e amodem/ --force-reinstall".split(" "))

    print("Verifying installation...")
    try:
        subprocess.check_call([str(venv_python), "-m", "amodem", "--help"], stdout=subprocess.DEVNULL)
        print("Verification passed: amodem is installed and runnable.")
    except subprocess.CalledProcessError:
        print("Verification failed: amodem returned non-zero exit code.")
        sys.exit(1)

    print("\nDone! Activate with:")
    if sys.platform == "win32":
        print(f"  {venv_dir}\\Scripts\\activate")
    else:
        print(f"  source {venv_dir}/bin/activate")
    print("\nRun directly with:")
    if sys.platform == "win32":
        print(f"  {venv_dir}\\Scripts\\python -m amodem ...")
    else:
        print(f"  source {venv_dir}/bin/python -m amodem ...")

    if platform.system() == "Windows":
        print("You might want to use -i/-o because audio devices might not work.")

if __name__ == "__main__":
    main()
