#!/usr/bin/env python3
"""Script to set up ASV machine configuration non-interactively."""

import subprocess
import sys


def setup_asv_machine():
    """Set up ASV machine configuration."""
    # Machine configuration answers
    answers = [
        "FR-IWL-MCINTOS1",  # machine name
        "Windows 11",  # os
        "AMD64",  # arch
        "11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz",  # cpu
        "8",  # num_cpu
        "16GB",  # ram
    ]

    # Run asv machine with answers piped in
    process = subprocess.Popen(
        [sys.executable, "-m", "asv", "machine", "--yes"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Send all answers
    input_data = "\n".join(answers) + "\n"
    stdout, stderr = process.communicate(input=input_data)

    print("ASV machine setup output:")
    print(stdout)
    if stderr:
        print("Errors/warnings:")
        print(stderr)

    return process.returncode == 0


if __name__ == "__main__":
    success = setup_asv_machine()
    if success:
        print("ASV machine setup completed successfully!")
    else:
        print("ASV machine setup failed!")
        sys.exit(1)
