# Remote scripts for execution at facilities.
#
# These scripts are designed to run on remote systems via SSH with minimal
# dependencies (Python 3.8+ stdlib only). They are loaded and executed
# using the run_python_script() / async_run_python_script() functions
# from executor.py.
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
# - extract_tdi_functions.py: TDI .fun file metadata extraction
# - extract_tdi_signals.py: TDI signal extraction with runtime probing
# - enumerate_ppf.py: JET PPF DDA/Dtype enumeration via ppf library
# - check_ppf.py: JET PPF signal validation via ppfdata()
# - enumerate_edas.py: JT-60SA EDAS category/data name enumeration
# - check_edas.py: JT-60SA EDAS signal validation via eddbreadOne()
