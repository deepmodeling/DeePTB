import argparse
import subprocess
import logging
import os
import sys
import shutil
from pathlib import Path
from dptb.postprocess.unified.system import TBSystem

log = logging.getLogger(__name__)

def pdso(
        INPUT: str,
        init_model: str = None,
        structure: str = None,
        data_dir: str = None,
        output_dir: str = "./",
        log_level: int = 20,
        log_path: str = None,
        ill_project: bool = True,
        ill_threshold: float = 5e-4,
        **kwargs
        ):
    """
    Run the DeePTB-Pardiso workflow (Export + Julia Backend).
    
    Parameters
    ----------
    INPUT : str
        Configuration JSON file path.
    init_model : str, optional
        Path to model checkpoint (.pth file).
    structure : str, optional
        Path to structure file (.vasp, .cif, etc.).
    data_dir : str, optional
        Directory containing pre-exported data (for run-only mode).
    output_dir : str
        Output directory for exported data and results.
    log_level : int
        Logging level (default: 20 = INFO).
    log_path : str, optional
        Path to log file.
    ill_project : bool
        Enable ill-conditioned state projection (default: True).
    ill_threshold : float
        Threshold for ill-conditioning detection (default: 5e-4).
    
    Modes:
    1. Full Flow: init_model + structure -> Export -> Julia
    2. Run Only: data_dir -> Julia
    """
    
    # 1. Determine Paths
    config_path = os.path.abspath(INPUT)
    output_path = os.path.abspath(output_dir)
    results_path = os.path.join(output_path, "results")
    
    # Ensure output directories exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    # 2. Export Phase (if init_model provided)
    if init_model:
        if not structure:
            log.error("Structure file (-stu) is required when using model checkpoint (-i).")
            sys.exit(1)
            
        log.info(f"Initializing TBSystem from {init_model} and {structure}...")
        
        try:
            # Initialize System
            tbsys = TBSystem(data=structure, calculator=init_model)
            
            # Export Data
            log.info(f"Exporting data to {output_path}...")
            tbsys.to_pardiso(output_dir=output_path)
            
            # Set input_dir to the exported directory
            input_data_dir = output_path
            
        except Exception as e:
            log.exception(f"Export failed: {e}")
            sys.exit(1)
            
    # 3. Run Only Phase (if data_dir provided)
    elif data_dir:
        input_data_dir = os.path.abspath(data_dir)
        if not os.path.exists(input_data_dir):
            log.error(f"Data directory not found: {input_data_dir}")
            sys.exit(1)
    else:
        log.error("Either model checkpoint (-i) or data directory (-d) must be provided.")
        sys.exit(1)

    # 4. Julia Backend Execution
    # Locate main.jl
    current_dir = Path(__file__).resolve().parent
    julia_script = current_dir.parent / "postprocess" / "pardiso" / "main.jl"

    if not julia_script.exists():
        log.error(f"Julia backend script not found at {julia_script}")
        sys.exit(1)

    # Prepare Command
    cmd = [
        "julia",
        str(julia_script),
        "--input_dir", input_data_dir,
        "--output_dir", results_path,
        "--config", config_path,
        "--ill_project", str(ill_project).lower(),
        "--ill_threshold", str(ill_threshold)
    ]

    log.info(f"Running Julia backend: {' '.join(cmd)}")
    log.info("-" * 60)
    
    try:
        # Stream output to console
        subprocess.run(cmd, check=True)
        log.info("-" * 60)
        log.info(f"Pardiso workflow completed. Results in: {results_path}")
    except subprocess.CalledProcessError as e:
        log.error(f"Julia execution failed with exit code {e.returncode}")
        # Don't exit here, let python cleanup if needed, or re-raise
        sys.exit(e.returncode)
    except FileNotFoundError:
        log.error("Julia executable not found. Please install Julia and ensure it is in your PATH.")
        sys.exit(1)
