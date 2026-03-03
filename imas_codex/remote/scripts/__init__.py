# Remote scripts for execution at facilities.
#
# Two execution paths with different Python version constraints:
#
# 1. run_python_script() / async_run_python_script() — uses the imas-codex
#    venv Python (3.12+). Scripts dispatched via this path may use modern
#    Python syntax and import from the venv's site-packages (e.g. MDSplus).
#
# 2. SSHWorkerPool / pooled_run_python_script() — uses /usr/bin/python3
#    (system Python, as low as 3.9). Scripts dispatched via exec() inside
#    the worker MUST be Python 3.9+ compatible and stdlib-only.
#
# Each script declares its minimum version in its docstring header:
#   - "Python 3.8+" — stdlib-only, safe for both execution paths
#   - "Python 3.12+" — requires venv, must ONLY be dispatched via
#     run_python_script() / async_run_python_script()
#
# All remote execution MUST use these scripts via the executor functions.
# Never inline SSH subprocess calls in scanner or discovery code.
#
# Scripts in this package:
# - scan_directories.py: Fast directory enumeration for discovery pipeline
# - enrich_directories.py: Directory metadata enrichment
# - get_user_info.py: User/group information extraction
# - check_signals.py: Batched MDSplus signal validation
# - check_signals_batch.py: High-throughput batched signal validation
# - extract_tdi_signals.py: TDI signal extraction with runtime probing
# - enumerate_ppf.py: JET PPF DDA/Dtype enumeration via ppf library
# - check_ppf.py: JET PPF signal validation via ppfdata()
# - enumerate_edas.py: JT-60SA EDAS category/data name enumeration
# - check_edas.py: JT-60SA EDAS signal validation via eddbreadOne()
