# Remote scripts for execution at facilities.
#
# These scripts are designed to run on remote systems via SSH with minimal
# dependencies (Python 3.8+ stdlib only). They are loaded and executed
# using the run_python_script() function from executor.py.
#
# Scripts in this package:
# - scan_directories.py: Fast directory enumeration for discovery pipeline
# - enrich_directories.py: Directory metadata enrichment
# - get_user_info.py: User/group information extraction
# - check_signals.py: Batched MDSplus signal validation
