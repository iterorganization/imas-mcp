#!/usr/bin/env python3
"""
Model comparison study for directory path scoring.

Evaluates different LLM models on their ability to score discovered paths
for the discovery pipeline. Tests:
- Claude 4.5 family: Haiku, Sonnet, Opus
- Gemini 3 family: Flash, Pro

Metrics evaluated:
- Score accuracy (does the score reflect the path's actual value?)
- Description quality (is the description accurate and informative?)
- Cost efficiency (tokens/$ per path)
- Latency (time per batch)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test directory data - simulates real scan results
# These represent realistic paths from fusion facilities (EPFL, ITER, JET)
# Expanded to 35 test cases including container directories
TEST_DIRECTORIES = [
    # =========================================================================
    # CONTAINERS (5 cases) - Score = exploration potential
    # =========================================================================
    {
        "path": "/work/imas",
        "total_files": 0,
        "total_dirs": 7,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {},
        "patterns_detected": ["IMAS"],
        "child_names": [
            "core/",
            "shared/",
            "extra/",
            "opt/",
            "etc/",
        ],
        "expected_score_min": 0.8,
        "expected_purpose": "container",
        "reason": "IMAS installation root - very high exploration potential",
    },
    {
        "path": "/work/projects",
        "total_files": 0,
        "total_dirs": 15,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {},
        "patterns_detected": [],
        "child_names": [
            "equilibrium/",
            "transport/",
            "heating/",
            "diagnostics/",
            "analysis/",
        ],
        "expected_score_min": 0.7,
        "expected_purpose": "container",
        "reason": "Projects directory - likely valuable children",
    },
    {
        "path": "/home/codes",
        "total_files": 0,
        "total_dirs": 20,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {},
        "patterns_detected": [],
        "child_names": [
            "liuqe/",
            "chease/",
            "astra/",
            "jorek/",
            "postproc/",
        ],
        "expected_score_min": 0.7,
        "expected_purpose": "container",
        "reason": "Codes directory - high exploration potential",
    },
    {
        "path": "/home/wilson",
        "total_files": 10,
        "total_dirs": 12,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"py": 5, "txt": 5},
        "patterns_detected": [],
        "child_names": [
            "codes/",
            "projects/",
            "data/",
            "Documents/",
            "Downloads/",
            ".bashrc",
        ],
        "expected_score_min": 0.5,
        "expected_purpose": "container",
        "reason": "User home - medium exploration potential",
    },
    {
        "path": "/home/user/Downloads",
        "total_files": 150,
        "total_dirs": 10,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"pdf": 50, "zip": 30, "tar.gz": 20, "deb": 10, "misc": 40},
        "patterns_detected": [],
        "child_names": [
            "paper_2024.pdf",
            "slides.pptx",
            "software.zip",
            "package.deb",
            "random/",
        ],
        "expected_score_min": 0.0,
        "expected_purpose": "container",
        "reason": "Downloads folder - low exploration potential (skip subtree)",
    },
    # =========================================================================
    # MODELING CODE (5 cases) - Should score >= 0.7
    # =========================================================================
    {
        "path": "/home/codes/liuqe",
        "total_files": 45,
        "total_dirs": 8,
        "has_readme": True,
        "has_makefile": True,
        "has_git": True,
        "file_type_counts": {"py": 30, "f90": 10, "md": 3, "txt": 2},
        "patterns_detected": ["equilibrium", "IMAS", "write_ids"],
        "child_names": [
            "liuqe.py",
            "imas_interface.py",
            "equilibrium_solver.f90",
            "README.md",
            "Makefile",
            ".git",
            "tests/",
            "docs/",
        ],
        "expected_score_min": 0.8,
        "expected_purpose": "modeling_code",
        "reason": "Equilibrium reconstruction code with IMAS integration",
    },
    {
        "path": "/home/codes/transport/astra",
        "total_files": 120,
        "total_dirs": 15,
        "has_readme": True,
        "has_makefile": True,
        "has_git": False,
        "file_type_counts": {"f90": 80, "py": 25, "inc": 10, "cfg": 5},
        "patterns_detected": ["transport", "equilibrium"],
        "child_names": [
            "astra_main.f90",
            "transport_solve.f90",
            "plasma_profiles.py",
            "README.txt",
            "Makefile",
            "config/",
            "input/",
            "output/",
        ],
        "expected_score_min": 0.75,
        "expected_purpose": "modeling_code",
        "reason": "Transport code (ASTRA) for plasma simulations",
    },
    {
        "path": "/home/codes/chease",
        "total_files": 80,
        "total_dirs": 10,
        "has_readme": True,
        "has_makefile": True,
        "has_git": True,
        "file_type_counts": {"f90": 60, "py": 15, "txt": 5},
        "patterns_detected": ["equilibrium", "IMAS", "flux"],
        "child_names": [
            "chease.f90",
            "imas_interface.py",
            "equilibrium.f90",
            "README.md",
            "Makefile",
            ".git",
            "examples/",
        ],
        "expected_score_min": 0.8,
        "expected_purpose": "modeling_code",
        "reason": "CHEASE equilibrium solver with IMAS",
    },
    {
        "path": "/work/codes/jintrac",
        "total_files": 500,
        "total_dirs": 50,
        "has_readme": True,
        "has_makefile": True,
        "has_git": False,
        "file_type_counts": {"f90": 350, "py": 100, "inc": 50},
        "patterns_detected": ["transport", "IMAS", "plasma", "equilibrium"],
        "child_names": [
            "jintrac_main.f90",
            "transport.f90",
            "imas_wrapper.py",
            "README.md",
            "Makefile",
            "modules/",
            "config/",
        ],
        "expected_score_min": 0.85,
        "expected_purpose": "modeling_code",
        "reason": "JINTRAC integrated modeling with IMAS",
    },
    {
        "path": "/home/codes/stability/jorek",
        "total_files": 400,
        "total_dirs": 35,
        "has_readme": True,
        "has_makefile": True,
        "has_git": True,
        "file_type_counts": {"f90": 300, "cpp": 50, "py": 50},
        "patterns_detected": ["MHD", "equilibrium", "stability"],
        "child_names": [
            "jorek_main.f90",
            "mhd_solver.f90",
            "stability.f90",
            "README.md",
            "Makefile",
            ".git",
            "input/",
        ],
        "expected_score_min": 0.8,
        "expected_purpose": "modeling_code",
        "reason": "JOREK MHD stability code",
    },
    # =========================================================================
    # DIAGNOSTIC CODE (5 cases) - Should score >= 0.5
    # =========================================================================
    {
        "path": "/home/diagnostics/thomson",
        "total_files": 50,
        "total_dirs": 8,
        "has_readme": True,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"py": 35, "idl": 10, "cfg": 5},
        "patterns_detected": ["MDSplus", "thomson"],
        "child_names": [
            "thomson_analysis.py",
            "ne_te_profiles.py",
            "calibration.py",
            "README.txt",
            "config/",
            "data/",
        ],
        "expected_score_min": 0.5,
        "expected_purpose": "diagnostic_code",
        "reason": "Thomson scattering diagnostic code",
    },
    {
        "path": "/home/diagnostics/bolometry",
        "total_files": 40,
        "total_dirs": 6,
        "has_readme": True,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"py": 25, "idl": 10, "cfg": 5},
        "patterns_detected": ["MDSplus", "radiation"],
        "child_names": [
            "bolometer_analysis.py",
            "radiation_profiles.py",
            "geometry.py",
            "README.txt",
            "calibration/",
            "data/",
        ],
        "expected_score_min": 0.5,
        "expected_purpose": "diagnostic_code",
        "reason": "Bolometry diagnostic analysis",
    },
    {
        "path": "/home/diagnostics/soft_xray",
        "total_files": 45,
        "total_dirs": 7,
        "has_readme": True,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"py": 30, "c": 10, "cfg": 5},
        "patterns_detected": ["MDSplus", "sxr"],
        "child_names": [
            "sxr_analysis.py",
            "tomography.py",
            "detector.c",
            "README.txt",
            "calibration/",
            "geometry/",
        ],
        "expected_score_min": 0.5,
        "expected_purpose": "diagnostic_code",
        "reason": "Soft X-ray diagnostic code",
    },
    {
        "path": "/home/diagnostics/interferometry",
        "total_files": 55,
        "total_dirs": 8,
        "has_readme": True,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"py": 40, "m": 10, "cfg": 5},
        "patterns_detected": ["MDSplus", "density", "interferometer"],
        "child_names": [
            "interferometer.py",
            "density_inversion.py",
            "geometry.py",
            "README.txt",
            "calibration/",
            "data/",
        ],
        "expected_score_min": 0.5,
        "expected_purpose": "diagnostic_code",
        "reason": "Interferometry diagnostic code",
    },
    {
        "path": "/home/diagnostics/ece",
        "total_files": 60,
        "total_dirs": 8,
        "has_readme": True,
        "has_makefile": False,
        "has_git": True,
        "file_type_counts": {"py": 45, "m": 10, "cfg": 5},
        "patterns_detected": ["MDSplus", "ECE", "temperature"],
        "child_names": [
            "ece_analysis.py",
            "temperature_profiles.py",
            "calibration.py",
            "README.md",
            ".git",
            "config/",
        ],
        "expected_score_min": 0.5,
        "expected_purpose": "diagnostic_code",
        "reason": "ECE diagnostic for electron temperature",
    },
    # =========================================================================
    # DATA INTERFACE (5 cases) - Should score >= 0.6
    # =========================================================================
    {
        "path": "/home/codes/mdsplus_tools",
        "total_files": 25,
        "total_dirs": 3,
        "has_readme": True,
        "has_makefile": False,
        "has_git": True,
        "file_type_counts": {"py": 20, "md": 2, "json": 3},
        "patterns_detected": ["MDSplus", "TreeNode"],
        "child_names": [
            "mds_reader.py",
            "tree_walker.py",
            "tdi_utils.py",
            "README.md",
            ".git",
            "tests/",
            "examples/",
        ],
        "expected_score_min": 0.7,
        "expected_purpose": "data_interface",
        "reason": "MDSplus data access utilities",
    },
    {
        "path": "/home/codes/eqdsk_tools",
        "total_files": 30,
        "total_dirs": 4,
        "has_readme": True,
        "has_makefile": False,
        "has_git": True,
        "file_type_counts": {"py": 25, "md": 3, "json": 2},
        "patterns_detected": ["equilibrium", "eqdsk", "psi"],
        "child_names": [
            "eqdsk_reader.py",
            "eqdsk_writer.py",
            "flux_functions.py",
            "README.md",
            ".git",
            "tests/",
        ],
        "expected_score_min": 0.7,
        "expected_purpose": "data_interface",
        "reason": "EQDSK file format tools for equilibrium",
    },
    {
        "path": "/home/codes/ids_tools",
        "total_files": 60,
        "total_dirs": 8,
        "has_readme": True,
        "has_makefile": False,
        "has_git": True,
        "file_type_counts": {"py": 50, "md": 5, "yaml": 5},
        "patterns_detected": ["IMAS", "ids", "read_ids", "write_ids"],
        "child_names": [
            "ids_utils.py",
            "ids_reader.py",
            "ids_writer.py",
            "README.md",
            ".git",
            "tests/",
            "examples/",
        ],
        "expected_score_min": 0.8,
        "expected_purpose": "data_interface",
        "reason": "IDS manipulation tools for IMAS",
    },
    {
        "path": "/home/codes/imas_python",
        "total_files": 80,
        "total_dirs": 10,
        "has_readme": True,
        "has_makefile": False,
        "has_git": True,
        "file_type_counts": {"py": 70, "md": 5, "yaml": 5},
        "patterns_detected": ["IMAS", "ids", "put_slice", "get_slice"],
        "child_names": [
            "imas_utils.py",
            "ids_wrapper.py",
            "data_entry.py",
            "README.md",
            ".git",
            "tests/",
        ],
        "expected_score_min": 0.8,
        "expected_purpose": "data_interface",
        "reason": "Python IMAS interface library",
    },
    {
        "path": "/home/codes/hdf5_tools",
        "total_files": 20,
        "total_dirs": 3,
        "has_readme": True,
        "has_makefile": False,
        "has_git": True,
        "file_type_counts": {"py": 18, "md": 2},
        "patterns_detected": ["HDF5", "h5py"],
        "child_names": [
            "hdf5_reader.py",
            "hdf5_writer.py",
            "README.md",
            ".git",
            "tests/",
        ],
        "expected_score_min": 0.6,
        "expected_purpose": "data_interface",
        "reason": "HDF5 data access utilities",
    },
    # =========================================================================
    # WORKFLOW (3 cases) - Should score 0.4-0.6
    # =========================================================================
    {
        "path": "/home/user/analysis/shot_review",
        "total_files": 20,
        "total_dirs": 3,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"py": 15, "ipynb": 5},
        "patterns_detected": ["MDSplus"],
        "child_names": [
            "shot_summary.py",
            "quick_look.py",
            "analysis.ipynb",
            "notebooks/",
            "output/",
        ],
        "expected_score_min": 0.4,
        "expected_purpose": "workflow",
        "reason": "Shot review workflow scripts",
    },
    {
        "path": "/home/codes/postprocessing",
        "total_files": 25,
        "total_dirs": 4,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"py": 20, "sh": 5},
        "patterns_detected": [],
        "child_names": [
            "convert_hdf5.py",
            "extract_data.py",
            "batch_process.sh",
            "output/",
            "temp/",
        ],
        "expected_score_min": 0.4,
        "expected_purpose": "workflow",
        "reason": "Batch processing workflow scripts",
    },
    {
        "path": "/home/codes/automation",
        "total_files": 30,
        "total_dirs": 5,
        "has_readme": True,
        "has_makefile": False,
        "has_git": True,
        "file_type_counts": {"py": 20, "sh": 8, "yaml": 2},
        "patterns_detected": [],
        "child_names": [
            "run_analysis.py",
            "submit_job.sh",
            "pipeline.yaml",
            "README.md",
            ".git",
            "logs/",
        ],
        "expected_score_min": 0.4,
        "expected_purpose": "workflow",
        "reason": "Automation and pipeline scripts",
    },
    # =========================================================================
    # VISUALIZATION (2 cases) - Should score 0.4-0.6
    # =========================================================================
    {
        "path": "/home/user/scripts/plotting",
        "total_files": 35,
        "total_dirs": 5,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"py": 30, "png": 5},
        "patterns_detected": [],
        "child_names": [
            "plot_profiles.py",
            "viz_equilibrium.py",
            "matplotlib_utils.py",
            "color_schemes.py",
            "output/",
            "figures/",
        ],
        "expected_score_min": 0.4,
        "expected_purpose": "visualization",
        "reason": "Plotting utilities for plasma data",
    },
    {
        "path": "/home/codes/visualize",
        "total_files": 40,
        "total_dirs": 6,
        "has_readme": True,
        "has_makefile": False,
        "has_git": True,
        "file_type_counts": {"py": 35, "png": 5},
        "patterns_detected": ["equilibrium"],
        "child_names": [
            "plot_equilibrium.py",
            "plot_profiles.py",
            "3d_viewer.py",
            "README.md",
            ".git",
            "examples/",
        ],
        "expected_score_min": 0.5,
        "expected_purpose": "visualization",
        "reason": "Equilibrium and profile visualization tools",
    },
    # =========================================================================
    # SIMULATION DATA (2 cases) - Should score 0.3-0.5
    # =========================================================================
    {
        "path": "/scratch/user/simulations/run_archive",
        "total_files": 10000,
        "total_dirs": 500,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"h5": 5000, "dat": 3000, "nc": 2000},
        "patterns_detected": [],
        "child_names": [
            "run_001/",
            "run_002/",
            "run_003/",
            "archived/",
            "old/",
        ],
        "expected_score_min": 0.0,
        "expected_purpose": "simulation_data",
        "reason": "Simulation output archive - data not code",
    },
    {
        "path": "/work/data/jintrac_outputs",
        "total_files": 500,
        "total_dirs": 50,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"h5": 300, "nc": 150, "log": 50},
        "patterns_detected": ["JINTRAC"],
        "child_names": [
            "scenario_001/",
            "scenario_002/",
            "analysis/",
            "archive/",
        ],
        "expected_score_min": 0.0,
        "expected_purpose": "simulation_data",
        "reason": "JINTRAC simulation outputs - data files",
    },
    # =========================================================================
    # DOCUMENTATION (2 cases) - Should score 0.3-0.5
    # =========================================================================
    {
        "path": "/home/user/teaching/plasma_physics",
        "total_files": 60,
        "total_dirs": 10,
        "has_readme": True,
        "has_makefile": False,
        "has_git": True,
        "file_type_counts": {"ipynb": 30, "py": 20, "pdf": 10},
        "patterns_detected": ["plasma", "equilibrium"],
        "child_names": [
            "lecture1.ipynb",
            "lecture2.ipynb",
            "exercises/",
            "solutions/",
            "README.md",
            ".git",
        ],
        "expected_score_min": 0.4,
        "expected_purpose": "documentation",
        "reason": "Teaching materials and tutorials",
    },
    {
        "path": "/home/user/Documents/papers",
        "total_files": 200,
        "total_dirs": 30,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"pdf": 150, "tex": 30, "bib": 10, "docx": 10},
        "patterns_detected": [],
        "child_names": [
            "plasma_paper_2023.pdf",
            "tokamak_review.pdf",
            "references.bib",
            "drafts/",
            "submitted/",
        ],
        "expected_score_min": 0.0,
        "expected_purpose": "documentation",
        "reason": "Papers and documents collection",
    },
    # =========================================================================
    # ARCHIVE (2 cases) - Should score < 0.3 (skip subtree)
    # =========================================================================
    {
        "path": "/home/user/backup/old_projects",
        "total_files": 200,
        "total_dirs": 50,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"bak": 100, "old": 50, "py": 30, "txt": 20},
        "patterns_detected": [],
        "child_names": [
            "project1_backup/",
            "project2_old/",
            "temp/",
            "archive/",
            "old_scripts/",
            "deprecated/",
            "2019/",
            "2020/",
        ],
        "expected_score_min": 0.0,
        "expected_purpose": "archive",
        "reason": "Backup directory with old/deprecated content",
    },
    {
        "path": "/home/codes/deprecated",
        "total_files": 100,
        "total_dirs": 20,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"py": 50, "f90": 30, "txt": 20},
        "patterns_detected": [],
        "child_names": [
            "old_liuqe/",
            "chease_v1/",
            "archive/",
            "backup/",
        ],
        "expected_score_min": 0.0,
        "expected_purpose": "archive",
        "reason": "Deprecated code archive",
    },
    # =========================================================================
    # BUILD ARTIFACTS (2 cases) - Should score < 0.3 (skip subtree)
    # =========================================================================
    {
        "path": "/home/user/venv/lib/python3.11",
        "total_files": 5000,
        "total_dirs": 200,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"py": 4500, "pyc": 500},
        "patterns_detected": [],
        "child_names": [
            "site-packages/",
            "numpy/",
            "matplotlib/",
            "scipy/",
            "__pycache__/",
            "pip/",
            "setuptools/",
        ],
        "expected_score_min": 0.0,
        "expected_purpose": "build_artifact",
        "reason": "Virtual environment - not user code",
    },
    {
        "path": "/home/user/.cache/pip",
        "total_files": 2000,
        "total_dirs": 100,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"whl": 500, "tar.gz": 500, "json": 1000},
        "patterns_detected": [],
        "child_names": [
            "wheels/",
            "http/",
            "selfcheck/",
            "packages/",
        ],
        "expected_score_min": 0.0,
        "expected_purpose": "build_artifact",
        "reason": "Pip cache directory",
    },
    # =========================================================================
    # SYSTEM (2 cases) - Should score < 0.3 (skip subtree)
    # =========================================================================
    {
        "path": "/var/log/plasma_sim",
        "total_files": 1000,
        "total_dirs": 10,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"log": 900, "txt": 100},
        "patterns_detected": [],
        "child_names": [
            "run_001.log",
            "run_002.log",
            "error.log",
            "debug.log",
            "2024-01/",
            "2024-02/",
            "archive/",
        ],
        "expected_score_min": 0.0,
        "expected_purpose": "system",
        "reason": "Log directory - no code value",
    },
    {
        "path": "/opt/modules/compilers",
        "total_files": 300,
        "total_dirs": 50,
        "has_readme": False,
        "has_makefile": False,
        "has_git": False,
        "file_type_counts": {"so": 200, "a": 50, "h": 50},
        "patterns_detected": [],
        "child_names": [
            "gcc-12.1/",
            "intel-2023/",
            "nvidia-hpc/",
            "lib/",
            "include/",
        ],
        "expected_score_min": 0.0,
        "expected_purpose": "system",
        "reason": "System compiler modules",
    },
]

# Models to test
MODELS_TO_TEST = [
    ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5"),
    ("anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5"),
    ("anthropic/claude-opus-4.5", "Claude Opus 4.5"),
    ("google/gemini-3-flash-preview", "Gemini 3 Flash"),
    ("google/gemini-3-pro-preview", "Gemini 3 Pro"),
]


@dataclass
class ModelResult:
    """Results for a single model evaluation."""

    model_id: str
    model_name: str
    total_time: float = 0.0
    total_cost: float = 0.0
    total_tokens: int = 0
    scored_dirs: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def score_accuracy(self) -> float:
        """Fraction of paths scored within expected range."""
        if not self.scored_dirs:
            return 0.0
        correct = 0
        for sd, expected in self.scored_dirs:
            expected_min = expected["expected_score_min"]
            # Consider correct if:
            # - High value (>= 0.7 expected) scores >= 0.6
            # - Medium value (0.4-0.7 expected) scores 0.3-0.8
            # - Low value (< 0.4 expected) scores < 0.5
            if expected_min >= 0.7:
                if sd.score >= 0.6:
                    correct += 1
            elif expected_min >= 0.4:
                if 0.3 <= sd.score <= 0.8:
                    correct += 1
            else:
                if sd.score < 0.5:
                    correct += 1
        return correct / len(self.scored_dirs)

    @property
    def purpose_accuracy(self) -> float:
        """Fraction of paths with correctly classified purpose."""
        if not self.scored_dirs:
            return 0.0
        correct = 0
        for sd, expected in self.scored_dirs:
            if sd.path_purpose.value == expected["expected_purpose"]:
                correct += 1
        return correct / len(self.scored_dirs)

    @property
    def cost_per_path(self) -> float:
        """USD cost per path scored."""
        if not self.scored_dirs:
            return 0.0
        return self.total_cost / len(self.scored_dirs)

    @property
    def time_per_path(self) -> float:
        """Seconds per path scored."""
        if not self.scored_dirs:
            return 0.0
        return self.total_time / len(self.scored_dirs)


def run_model_evaluation(
    model_id: str, model_name: str, directories: list[dict]
) -> ModelResult:
    """Evaluate a single model on the test directories."""
    from imas_codex.discovery.scorer import DirectoryScorer

    result = ModelResult(model_id=model_id, model_name=model_name)

    try:
        scorer = DirectoryScorer(model=model_id)

        start = time.time()
        batch = scorer.score_batch(
            directories=directories,
            focus="fusion physics analysis codes with IMAS integration",
            threshold=0.7,
        )
        elapsed = time.time() - start

        result.total_time = elapsed
        result.total_cost = batch.total_cost
        result.total_tokens = batch.tokens_used

        # Match scored dirs with expected values
        for sd in batch.scored_dirs:
            for expected in directories:
                if expected["path"] == sd.path:
                    result.scored_dirs.append((sd, expected))
                    break

    except Exception as e:
        result.errors.append(str(e))

    return result


def print_detailed_results(result: ModelResult):
    """Print detailed per-path results for a model."""
    print(f"\n{'=' * 80}")
    print(f"Model: {result.model_name} ({result.model_id})")
    print(f"{'=' * 80}")

    if result.errors:
        print(f"ERRORS: {result.errors}")
        return

    print(
        f"Time: {result.total_time:.1f}s | Cost: ${result.total_cost:.4f} | Tokens: {result.total_tokens}"
    )
    print(
        f"Score Accuracy: {result.score_accuracy:.1%} | Purpose Accuracy: {result.purpose_accuracy:.1%}"
    )
    print()

    # Table header
    print(f"{'Path':<45} {'Score':>6} {'Expected':>8} {'Purpose':<15} {'OK?':>4}")
    print("-" * 80)

    for sd, expected in result.scored_dirs:
        path_short = sd.path if len(sd.path) <= 45 else "..." + sd.path[-42:]
        expected_min = expected["expected_score_min"]
        expected_purpose = expected["expected_purpose"]

        # Determine if scoring was correct
        if expected_min >= 0.7:
            ok = "✓" if sd.score >= 0.6 else "✗"
        elif expected_min >= 0.4:
            ok = "✓" if 0.3 <= sd.score <= 0.8 else "✗"
        else:
            ok = "✓" if sd.score < 0.5 else "✗"

        purpose_ok = "✓" if sd.path_purpose.value == expected_purpose else ""

        print(
            f"{path_short:<45} {sd.score:>6.2f} {expected_min:>8.2f} {sd.path_purpose.value:<15} {ok}{purpose_ok}"
        )

    print()
    print("Descriptions:")
    print("-" * 40)
    for sd, expected in result.scored_dirs:
        path_short = expected["path"].split("/")[-1]
        print(f"{path_short}: {sd.description[:70]}...")


def print_summary_table(results: list[ModelResult]):
    """Print summary comparison table."""
    print("\n" + "=" * 100)
    print("SUMMARY COMPARISON")
    print("=" * 100)
    print()

    # Header
    print(
        f"{'Model':<25} {'Score Acc':>10} {'Purpose Acc':>12} {'Time (s)':>10} {'Cost ($)':>10} {'$/path':>10}"
    )
    print("-" * 100)

    for r in results:
        if r.errors:
            print(f"{r.model_name:<25} ERROR: {r.errors[0][:50]}")
        else:
            print(
                f"{r.model_name:<25} {r.score_accuracy:>10.1%} {r.purpose_accuracy:>12.1%} "
                f"{r.total_time:>10.1f} {r.total_cost:>10.4f} {r.cost_per_path:>10.5f}"
            )

    print()

    # Recommendation
    valid_results = [r for r in results if not r.errors]
    if valid_results:
        # Sort by score accuracy, then by cost efficiency
        best_accuracy = max(valid_results, key=lambda r: r.score_accuracy)
        best_value = max(
            valid_results, key=lambda r: r.score_accuracy / max(r.cost_per_path, 0.0001)
        )
        cheapest = min(valid_results, key=lambda r: r.cost_per_path)

        print("RECOMMENDATIONS:")
        print(
            f"  Best Accuracy:  {best_accuracy.model_name} ({best_accuracy.score_accuracy:.1%})"
        )
        print(f"  Best Value:     {best_value.model_name} (acc/cost ratio)")
        print(
            f"  Cheapest:       {cheapest.model_name} (${cheapest.cost_per_path:.5f}/path)"
        )


def main():
    """Run the model comparison study."""
    # Check API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set")
        print("Run: source .env")
        sys.exit(1)

    print("=" * 80)
    print("MODEL COMPARISON STUDY: Directory Path Scoring")
    print("=" * 80)
    print(f"Test directories: {len(TEST_DIRECTORIES)}")
    print(f"Models to test: {len(MODELS_TO_TEST)}")
    print()

    results = []

    for model_id, model_name in MODELS_TO_TEST:
        print(f"\nTesting {model_name}...")
        result = run_model_evaluation(model_id, model_name, TEST_DIRECTORIES)
        results.append(result)

        if result.errors:
            print(f"  ERROR: {result.errors[0]}")
        else:
            print(f"  Score accuracy: {result.score_accuracy:.1%}")
            print(f"  Purpose accuracy: {result.purpose_accuracy:.1%}")
            print(f"  Time: {result.total_time:.1f}s, Cost: ${result.total_cost:.4f}")

    # Print detailed results
    for result in results:
        print_detailed_results(result)

    # Print summary
    print_summary_table(results)

    # Return results for programmatic use
    return results


if __name__ == "__main__":
    main()
