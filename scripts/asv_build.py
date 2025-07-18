#!/usr/bin/env python3
"""
ASV build script that handles all dependencies and environment setup.
"""

import os
import subprocess
import sys


def main():
    """Build the wheel for ASV with proper environment setup."""
    # Set the IDS filter for fast builds
    os.environ["IDS_FILTER"] = "core_profiles,equilibrium"

    # Get build cache directory and build directory from command line args
    if len(sys.argv) < 3:
        print(
            "Error: Build cache directory and build directory not provided",
            file=sys.stderr,
        )
        sys.exit(1)

    build_cache_dir = sys.argv[1]
    build_dir = sys.argv[2]

    try:
        # First, install build dependencies
        print("Installing build dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "build", "hatchling", "hatch-vcs"],
            check=True,
        )

        # Build the wheel
        print(f"Building wheel from {build_dir} to {build_cache_dir}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                build_cache_dir,
                build_dir,
            ],
            check=True,
        )

        print("Build completed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
