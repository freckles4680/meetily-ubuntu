#!/usr/bin/env python3
"""Check that this machine can run the Meetily CUDA (NVIDIA GPU) package.

Reports the NVIDIA driver and GPU, and whether the installed Meetily binaries
find every shared library they need. Needs nothing beyond Python 3 and the
NVIDIA driver; it changes nothing on the system.

Usage: python3 scripts/check_nvidia_gpu.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

MIN_DRIVER = 570
# Default architectures of the Ubuntu Package workflow; 12.0 also ships PTX.
BUILT_FOR = [(7, 5), (8, 0), (8, 6), (8, 9), (12, 0)]
PTX_FROM = (12, 0)
BINARIES = [Path("/usr/bin/meetily"), Path("/usr/bin/llama-helper")]
CUDA_LIBS = Path("/usr/lib/meetily/cuda")


def parse_cap(cap):
    major, minor = cap.split(".")
    return int(major), int(minor)


def covered(cap):
    # GPU code runs on the same major version with an equal or higher minor;
    # PTX is compiled by the driver for any newer GPU.
    c = parse_cap(cap)
    return any(c[0] == b[0] and c[1] >= b[1] for b in BUILT_FOR) or c >= PTX_FROM


def gpu_code_note(cap):
    c = parse_cap(cap)
    if c in BUILT_FOR:
        return f"package has GPU code built for compute capability {cap}"
    if covered(cap):
        return f"package GPU code is compatible with compute capability {cap}"
    return f"package is not built for compute capability {cap}; rebuild with it in cuda_architectures"


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    ok = True

    if not shutil.which("nvidia-smi"):
        print("FAIL  nvidia-smi not found: install the NVIDIA driver (sudo ubuntu-drivers install), then reboot.")
        return 1
    res = run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total", "--format=csv,noheader"])
    if res.returncode != 0:
        print(f"FAIL  nvidia-smi failed (driver not loaded? reboot after installing):\n{res.stderr.strip()}")
        return 1
    for line in res.stdout.strip().splitlines():
        name, driver, cap, mem = [x.strip() for x in line.split(",")]
        major = int(driver.split(".")[0])
        print(f"GPU   {name}, {mem}, compute capability {cap}, driver {driver}")
        if major < MIN_DRIVER:
            print(f"WARN  driver {driver} is older than {MIN_DRIVER}; update it if Meetily fails to use the GPU.")
        print("OK    " + gpu_code_note(cap) if covered(cap) else "FAIL  " + gpu_code_note(cap))
        ok = ok and covered(cap)

    if not CUDA_LIBS.is_dir():
        print(f"INFO  {CUDA_LIBS} not found: the CUDA package is not installed (or the CPU package is).")
    for binary in BINARIES:
        if not binary.exists():
            print(f"INFO  {binary} not installed")
            continue
        res = run(["ldd", str(binary)])
        missing = [l.split()[0] for l in res.stdout.splitlines() if "not found" in l]
        if missing:
            ok = False
            print(f"FAIL  {binary} is missing libraries: {', '.join(missing)}")
        else:
            cuda = [l.split()[0] for l in res.stdout.splitlines() if "libcu" in l]
            print(f"OK    {binary} finds all its libraries" + (f" (CUDA: {', '.join(cuda)})" if cuda else " (no CUDA libraries: CPU build)"))

    print("\nAll good." if ok else "\nProblems found, see FAIL lines above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
