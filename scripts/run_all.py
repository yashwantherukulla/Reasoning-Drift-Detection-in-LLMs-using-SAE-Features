"""
scripts/run_all.py
==================
Orchestrator script to run the full Reasoning Drift Detection pipeline end-to-end.

Usage:
    uv run scripts/run_all.py --pilot   # Runs on 10 problems
    uv run scripts/run_all.py --full    # Runs on 100 problems
"""

import argparse
import subprocess
import sys
import time

def run_step(command: list[str], step_name: str) -> None:
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING: {step_name}")
    print(f"💻 COMMAND: {' '.join(command)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    try:
        # Run command and stream output directly to stdout/stderr
        result = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Step '{step_name}' failed with exit code {e.returncode}.")
        sys.exit(1)
        
    elapsed = time.time() - start_time
    print(f"\n✅ SUCCESS: '{step_name}' completed in {elapsed:.1f}s.\n")

def main():
    parser = argparse.ArgumentParser(description="Run the full Reasoning Drift Detection pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true", help="Run the pilot test (10 problems).")
    group.add_argument("--full", action="store_true", help="Run the full evaluation (100 problems).")
    args = parser.parse_args()
    
    n_problems = 100 if args.full else 10
    mode_str = "FULL (100 problems)" if args.full else "PILOT (10 problems)"
    
    print(f"Starting pipeline in {mode_str} mode...")
    
    steps = [
        ("Phase 1: Build Dataset", ["uv", "run", "scripts/build_dataset.py", f"dataset.n_problems={n_problems}"]),
        ("Phase 2: Inference", ["uv", "run", "scripts/run_inference.py", f"dataset.n_problems={n_problems}"]),
        ("Phase 3: SAE Projection", ["uv", "run", "scripts/run_sae_projection.py", f"dataset.n_problems={n_problems}"]),
        ("Phase 4: Base Analysis", ["uv", "run", "scripts/run_analysis.py", f"dataset.n_problems={n_problems}"]),
        ("Phase 4+: Extended Analysis", ["uv", "run", "scripts/run_extended_analysis.py", f"dataset.n_problems={n_problems}"]),
        ("Phase 5: Causal Patching", ["uv", "run", "scripts/run_patching.py", f"dataset.n_problems={n_problems}"]),
        ("Phase 6: Visualization", ["uv", "run", "scripts/run_visualization.py"])
    ]
    
    total_start = time.time()
    
    for name, cmd in steps:
        run_step(cmd, name)
        
    total_elapsed = time.time() - total_start
    print(f"{'='*80}")
    print(f"🎉 PIPELINE COMPLETE! Total time: {total_elapsed / 60:.1f} minutes.")
    print(f"Figures and outputs are available in 'results/' and 'data/'.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
