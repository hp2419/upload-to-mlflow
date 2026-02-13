#!/usr/bin/env python3
"""Simple script to upload MLPerf results to MLflow server."""

import argparse
import ast
import re
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import mlflow
    from mlflow.exceptions import MlflowException
except ImportError as e:
    if 'boto3' in str(e).lower() or 'No module named' in str(e):
        print("ERROR: Missing required dependency 'boto3'")
        print("MLflow requires boto3 for artifact storage.")
        print("\nTo fix this, install boto3:")
        print("  pip install boto3")
        print("\nOr if using conda:")
        print("  conda install boto3")
        sys.exit(1)
    else:
        print(f"ERROR: Failed to import mlflow: {e}")
        print("Please install mlflow: pip install mlflow")
        sys.exit(1)

# Check for boto3 early to provide a clear error message
try:
    import boto3  # noqa: F401
    boto3_available = True
except ImportError:
    boto3_available = False
    print("WARNING: boto3 is not installed. MLflow may require it for artifact storage.")
    print("If you encounter errors, install boto3 with: pip install boto3")
    print("Continuing anyway...\n")


def parse_mlperf_summary(summary_file: Path) -> tuple[dict, dict]:
    """Parse MLPerf summary file and extract all metrics.
    
    Args:
        summary_file: Path to mlperf_log_summary.txt
        
    Returns:
        Tuple of (metrics_dict, params_dict) containing all parsed metrics and parameters
    """
    metrics = {}
    params = {}
    
    if not summary_file.exists():
        print(f"Warning: Summary file not found: {summary_file}")
        return metrics, params
    
    content = summary_file.read_text()
    
    # Parse main results
    samples_per_sec_match = re.search(r'Samples per second:\s*([\d.]+)', content)
    if samples_per_sec_match:
        metrics['samples_per_second'] = float(samples_per_sec_match.group(1))
    
    result_match = re.search(r'Result is\s*:\s*(\w+)', content)
    if result_match:
        params['result_valid'] = result_match.group(1) == 'VALID'
    
    # Parse validation checks
    min_duration_match = re.search(r'Min duration satisfied\s*:\s*(\w+)', content)
    if min_duration_match:
        params['min_duration_satisfied'] = min_duration_match.group(1) == 'Yes'
    
    min_queries_match = re.search(r'Min queries satisfied\s*:\s*(\w+)', content)
    if min_queries_match:
        params['min_queries_satisfied'] = min_queries_match.group(1) == 'Yes'
    
    early_stopping_match = re.search(r'Early stopping satisfied:\s*(\w+)', content)
    if early_stopping_match:
        params['early_stopping_satisfied'] = early_stopping_match.group(1) == 'Yes'
    
    # Parse latency metrics (convert nanoseconds to seconds for readability)
    latency_patterns = {
        'min_latency_ns': r'Min latency \(ns\)\s*:\s*(\d+)',
        'max_latency_ns': r'Max latency \(ns\)\s*:\s*(\d+)',
        'mean_latency_ns': r'Mean latency \(ns\)\s*:\s*(\d+)',
        'p50_latency_ns': r'50\.00 percentile latency \(ns\)\s*:\s*(\d+)',
        'p90_latency_ns': r'90\.00 percentile latency \(ns\)\s*:\s*(\d+)',
        'p95_latency_ns': r'95\.00 percentile latency \(ns\)\s*:\s*(\d+)',
        'p97_latency_ns': r'97\.00 percentile latency \(ns\)\s*:\s*(\d+)',
        'p99_latency_ns': r'99\.00 percentile latency \(ns\)\s*:\s*(\d+)',
        'p99_9_latency_ns': r'99\.90 percentile latency \(ns\)\s*:\s*(\d+)',
    }
    
    for metric_name, pattern in latency_patterns.items():
        match = re.search(pattern, content)
        if match:
            ns_value = int(match.group(1))
            metrics[metric_name] = ns_value
            # Also store in seconds for easier reading
            metrics[metric_name.replace('_ns', '_s')] = ns_value / 1e9
    
    # Parse test parameters
    param_patterns = {
        'samples_per_query': r'samples_per_query\s*:\s*(\d+)',
        'target_qps': r'target_qps\s*:\s*([\d.]+)',
        'target_latency_ns': r'target_latency \(ns\):\s*(\d+)',
        'max_async_queries': r'max_async_queries\s*:\s*(\d+)',
        'min_duration_ms': r'min_duration \(ms\):\s*(\d+)',
        'max_duration_ms': r'max_duration \(ms\):\s*(\d+)',
        'min_query_count': r'min_query_count\s*:\s*(\d+)',
        'max_query_count': r'max_query_count\s*:\s*(\d+)',
        'qsl_rng_seed': r'qsl_rng_seed\s*:\s*(\d+)',
        'sample_index_rng_seed': r'sample_index_rng_seed\s*:\s*(\d+)',
        'schedule_rng_seed': r'schedule_rng_seed\s*:\s*(\d+)',
        'accuracy_log_rng_seed': r'accuracy_log_rng_seed\s*:\s*(\d+)',
        'accuracy_log_probability': r'accuracy_log_probability\s*:\s*([\d.]+)',
        'accuracy_log_sampling_target': r'accuracy_log_sampling_target\s*:\s*(\d+)',
        'performance_sample_count': r'performance_sample_count\s*:\s*(\d+)',
    }
    
    for param_name, pattern in param_patterns.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1)
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    params[param_name] = float(value)
                else:
                    params[param_name] = int(value)
            except ValueError:
                params[param_name] = value
    
    # Parse scenario and mode
    scenario_match = re.search(r'Scenario\s*:\s*(\w+)', content)
    if scenario_match:
        params['scenario'] = scenario_match.group(1)
    
    mode_match = re.search(r'Mode\s*:\s*(\w+)', content)
    if mode_match:
        params['mode'] = mode_match.group(1)
    
    sut_match = re.search(r'SUT name\s*:\s*(.+)', content)
    if sut_match:
        params['sut_name'] = sut_match.group(1).strip()
    
    return metrics, params


def parse_log_tags(folder: Path) -> dict:
    """Parse tags from log files in the directory.
    
    Looks for:
    - mlperf_log_summary.txt: scenario, mode, target_qps, etc.
    - mlperf_log_dynamo.vllm.stderr.rank0.txt: engine config with parallelism, quantization, etc.
    - mlperf_log_dynamo.vllm.stdout.rank0.txt: architecture, chunked prefill, etc.
    
    Args:
        folder: Path to the results folder containing log files
        
    Returns:
        Dictionary of tags extracted from log files
    """
    tags = {}
    
    # Parse from summary file for scenario, mode, target_qps
    summary_file = folder / "mlperf_log_summary.txt"
    if summary_file.exists():
        summary_content = summary_file.read_text()
        
        # Scenario
        scenario_match = re.search(r'Scenario\s*:\s*(\w+)', summary_content)
        if scenario_match:
            tags['scenario'] = scenario_match.group(1).lower()
        
        # Mode
        mode_match = re.search(r'Mode\s*:\s*(\w+)', summary_content)
        if mode_match:
            mode = mode_match.group(1)
            tags['mode'] = mode.lower()
            if mode == 'PerformanceOnly':
                tags['test_mode'] = 'performance_only'
            elif mode == 'AccuracyOnly':
                tags['test_mode'] = 'accuracy_only'
        
        # Target QPS
        target_qps_match = re.search(r'target_qps\s*:\s*([\d.]+)', summary_content)
        if target_qps_match:
            tags['target_qps'] = target_qps_match.group(1)
    
    # Parse from vLLM stderr file for engine configuration
    stderr_file = folder / "mlperf_log_dynamo.vllm.stderr.rank0.txt"
    if not stderr_file.exists():
        # Try alternative naming
        stderr_files = list(folder.glob("mlperf_log_dynamo.vllm.stderr.rank*.txt"))
        if stderr_files:
            stderr_file = stderr_files[0]
    
    stderr_content = ""
    if stderr_file.exists():
        stderr_content = stderr_file.read_text()
        
        # Parse engine config line - extract key parameters with regex
        # Pattern: tensor_parallel_size=4, pipeline_parallel_size=1, data_parallel_size=1
        tp_match = re.search(r'tensor_parallel_size=(\d+)', stderr_content)
        if tp_match:
            tags['vllm_tp'] = tp_match.group(1)
            tags['tp'] = tp_match.group(1)
        
        pp_match = re.search(r'pipeline_parallel_size=(\d+)', stderr_content)
        if pp_match:
            tags['vllm_pp'] = pp_match.group(1)
            tags['pp'] = pp_match.group(1)
        
        dp_match = re.search(r'data_parallel_size=(\d+)', stderr_content)
        if dp_match:
            tags['vllm_dp'] = dp_match.group(1)
            tags['dp'] = dp_match.group(1)
        
        # Quantization
        quant_match = re.search(r'quantization=([^,\s\)]+)', stderr_content)
        if quant_match:
            tags['vllm_quantization'] = quant_match.group(1)
        
        # Data type
        dtype_match = re.search(r'dtype=([^,\s\)]+)', stderr_content)
        if dtype_match:
            tags['vllm_dtype'] = dtype_match.group(1)
        
        # Max sequence length
        max_seq_match = re.search(r'max_seq_len=(\d+)', stderr_content)
        if max_seq_match:
            tags['vllm_max_seq_len'] = max_seq_match.group(1)
        
        # Enforce eager
        eager_match = re.search(r'enforce_eager=(\w+)', stderr_content)
        if eager_match:
            tags['vllm_enforce_eager'] = 'enabled' if eager_match.group(1) == 'True' else 'disabled'
        
        # Enable chunked prefill
        chunked_match = re.search(r'enable_chunked_prefill=(\w+)', stderr_content)
        if chunked_match:
            tags['vllm_chunked_prefill'] = 'enabled' if chunked_match.group(1) == 'True' else 'disabled'
        
        # Enable prefix caching
        prefix_match = re.search(r'enable_prefix_caching=(\w+)', stderr_content)
        if prefix_match:
            tags['vllm_prefix_caching'] = 'enabled' if prefix_match.group(1) == 'True' else 'disabled'
        
        # Expert parallelism
        if 'Expert parallelism is enabled' in stderr_content:
            tags['vllm_ep'] = 'enabled'
            tags['ep'] = 'enabled'
            # Extract expert count
            ep_match = re.search(r'Local/global number of experts:\s*(\d+)/(\d+)', stderr_content)
            if ep_match:
                tags['vllm_experts_local'] = ep_match.group(1)
                tags['vllm_experts_global'] = ep_match.group(2)
        
        # vLLM version
        version_match = re.search(r'Initializing a V1 LLM engine \(([^\)]+)\)', stderr_content)
        if version_match:
            tags['vllm_version'] = version_match.group(1)
        
        # Model path (extract model name)
        model_match = re.search(r"model='([^']+)'", stderr_content)
        if model_match:
            model_path = model_match.group(1)
            # Extract model name from path
            if 'Qwen3-VL' in model_path:
                tags['model'] = 'Qwen3-VL-235B-A22B'
            # Extract quantization from path if present
            if 'FP8' in model_path:
                tags['model_quantization'] = 'FP8'
    
    # Parse from vLLM stdout file for additional info
    stdout_file = folder / "mlperf_log_dynamo.vllm.stdout.rank0.txt"
    if not stdout_file.exists():
        # Try alternative naming
        stdout_files = list(folder.glob("mlperf_log_dynamo.vllm.stdout.rank*.txt"))
        if stdout_files:
            stdout_file = stdout_files[0]
    
    if stdout_file.exists():
        stdout_content = stdout_file.read_text()
        
        # Architecture
        arch_match = re.search(r'Resolved architecture:\s*(\w+)', stdout_content)
        if arch_match:
            tags['vllm_architecture'] = arch_match.group(1)
            tags['architecture'] = arch_match.group(1)
        
        # Max model len
        max_model_match = re.search(r'Using max model len\s+(\d+)', stdout_content)
        if max_model_match:
            tags['vllm_max_model_len'] = max_model_match.group(1)
        
        # Chunked prefill status
        chunked_status_match = re.search(r'Chunked prefill is (enabled|disabled)', stdout_content)
        if chunked_status_match:
            tags['vllm_chunked_prefill'] = chunked_status_match.group(1)
        
        # Chunked prefill max tokens
        chunked_tokens_match = re.search(r'max_num_batched_tokens=(\d+)', stdout_content)
        if chunked_tokens_match:
            tags['vllm_max_batched_tokens'] = chunked_tokens_match.group(1)
        
        # Asynchronous scheduling
        if 'Asynchronous scheduling is enabled' in stdout_content:
            tags['vllm_async_scheduling'] = 'enabled'
    
    # Parse scheduler config if available
    if stderr_content:
        scheduler_match = re.search(r"Scheduler config values:\s*\{'max_num_seqs':\s*(\d+),\s*'max_num_batched_tokens':\s*(\d+)\}", stderr_content)
        if scheduler_match:
            tags['vllm_max_num_seqs'] = scheduler_match.group(1)
            if not tags.get('vllm_max_batched_tokens'):
                tags['vllm_max_batched_tokens'] = scheduler_match.group(2)
    
    return tags


def find_run_directories(root_dir: Path) -> list[Path]:
    """Recursively find all directories that contain mlperf_log_summary.txt.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of directories containing MLPerf run results
    """
    run_dirs = []
    for path in root_dir.rglob("mlperf_log_summary.txt"):
        run_dirs.append(path.parent)
    return sorted(run_dirs)


def upload_to_mlflow(
    folder_path: str,
    mlflow_uri: str = "http://localhost:5000",
    experiment_name: str = "qwen3vl_benchmark",
    run_name: Optional[str] = None,
    include_trace: bool = True,
    exclude_trace_pid: bool = False,
    recursive: bool = True,
) -> None:
    """Upload a folder to MLflow server as artifacts.
    
    Args:
        folder_path: Path to the folder containing results to upload
        mlflow_uri: MLflow server URI
        experiment_name: Name of the MLflow experiment
        run_name: Name for the MLflow run (defaults to folder name)
        include_trace: Whether to include trace files (default: True)
        exclude_trace_pid: Whether to exclude trace_pid*.json files (default: False)
        recursive: If True and folder doesn't contain log files, search recursively for run directories (default: True)
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Set MLflow tracking URI (do this once at the start)
    try:
        mlflow.set_tracking_uri(mlflow_uri)
    except Exception as e:
        if 'boto3' in str(e).lower() or 'No module named' in str(e):
            print(f"ERROR: MLflow requires boto3 for artifact storage.")
            print(f"Error: {e}")
            print("\nTo fix this, install boto3:")
            print("  pip install boto3")
            raise
        else:
            raise
    
    # Set or create experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name}")
    except Exception:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print(f"Using existing experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    
    # Check if this directory contains log files directly
    has_logs = (folder / "mlperf_log_summary.txt").exists()
    
    # If no logs in current directory and recursive is enabled, find all run directories
    if not has_logs and recursive:
        print(f"Directory {folder_path} doesn't contain log files directly.")
        print("Searching recursively for run directories...")
        run_dirs = find_run_directories(folder)
        
        if not run_dirs:
            raise ValueError(f"No MLPerf run directories found in {folder_path}")
        
        print(f"Found {len(run_dirs)} run directory(ies). Processing each...")
        
        # Process each run directory
        for run_dir in run_dirs:
            # Generate run name from relative path
            rel_path = run_dir.relative_to(folder)
            if run_name:
                # Use provided run_name as prefix
                dir_run_name = f"{run_name}_{rel_path}"
            else:
                dir_run_name = str(rel_path).replace("/", "_")
            
            print(f"\n{'='*80}")
            print(f"Processing run directory: {run_dir}")
            print(f"Run name: {dir_run_name}")
            print(f"{'='*80}\n")
            
            # Process this run directory (with recursive=False to avoid infinite loop)
            _upload_single_run(
                folder=run_dir,
                run_name=dir_run_name,
                mlflow_uri=mlflow_uri,
                include_trace=include_trace,
                exclude_trace_pid=exclude_trace_pid,
            )
        
        print(f"\n{'='*80}")
        print(f"Successfully processed {len(run_dirs)} run directory(ies)")
        print(f"{'='*80}")
        return
    
    # Use folder name as run name if not provided
    if run_name is None:
        run_name = folder.name
    
    # Process single run
    _upload_single_run(
        folder=folder,
        run_name=run_name,
        mlflow_uri=mlflow_uri,
        include_trace=include_trace,
        exclude_trace_pid=exclude_trace_pid,
    )


def _upload_single_run(
    folder: Path,
    run_name: str,
    mlflow_uri: str,
    include_trace: bool = True,
    exclude_trace_pid: bool = False,
) -> None:
    """Upload a single run directory to MLflow.
    
    Args:
        folder: Path to the folder containing results to upload
        run_name: Name for the MLflow run
        mlflow_uri: MLflow server URI
        include_trace: Whether to include trace files
        exclude_trace_pid: Whether to exclude trace_pid*.json files
    """
    # Start a new run
    with mlflow.start_run(run_name=run_name):
        print(f"Starting MLflow run: {run_name}")
        
        # Parse and log metrics from summary file
        summary_file = folder / "mlperf_log_summary.txt"
        if summary_file.exists():
            print("Parsing metrics from mlperf_log_summary.txt...")
            metrics, params = parse_mlperf_summary(summary_file)
            
            if metrics:
                print(f"Logging {len(metrics)} metrics...")
                mlflow.log_metrics(metrics)
                for metric_name, value in metrics.items():
                    print(f"  - {metric_name}: {value}")
            
            if params:
                print(f"Logging {len(params)} parameters...")
                mlflow.log_params(params)
                for param_name, value in params.items():
                    print(f"  - {param_name}: {value}")
        else:
            print(f"Warning: Summary file not found: {summary_file}")
        
        # Parse and log tags from log files
        print("Parsing tags from log files...")
        tags = parse_log_tags(folder)
        
        if tags:
            print(f"Logging {len(tags)} tags from log files...")
            mlflow.set_tags(tags)
            for tag_name, value in tags.items():
                print(f"  - {tag_name}: {value}")
        else:
            print("Warning: No tags found in log files")
        
        # Log all files in the folder as artifacts
        all_files = [f for f in folder.iterdir() if f.is_file()]
        print(f"Found {len(all_files)} total files in directory")
        
        # Filter trace files - skip large trace files to avoid upload issues
        # Trace files are typically very large and may cause upload issues
        files = []
        skipped_trace_files = []
        skipped_trace_pid_files = []
        large_file_threshold_mb = 50  # Skip trace files larger than 50MB
        
        for f in all_files:
            # Check if it's a trace_pid*.json file
            is_trace_pid_file = (
                f.name.startswith("trace_pid") and f.suffix == ".json"
            )
            
            # Check if it's a trace file
            is_trace_file = (
                f.name == "mlperf_log_trace.json" or
                (f.name.startswith("trace_") and f.suffix == ".json")
            )
            
            # Skip trace_pid*.json files if exclude_trace_pid is True
            if exclude_trace_pid and is_trace_pid_file:
                skipped_trace_pid_files.append(f.name)
                continue
            
            if is_trace_file:
                # Check file size for trace files
                try:
                    file_size_mb = f.stat().st_size / (1024 * 1024)
                    if file_size_mb > large_file_threshold_mb:
                        skipped_trace_files.append((f.name, file_size_mb))
                        continue  # Skip large trace files
                except Exception as e:
                    print(f"Warning: Could not check size of {f.name}: {e}")
                    # If we can't check size, skip it to be safe
                    skipped_trace_files.append((f.name, "unknown"))
                    continue
            
            # Include all non-trace files, and small trace files if include_trace is True
            if not is_trace_file or include_trace:
                files.append(f)
        
        if skipped_trace_pid_files:
            print(f"Skipped {len(skipped_trace_pid_files)} trace_pid*.json file(s):")
            for name in skipped_trace_pid_files:
                print(f"  - {name}")
        
        if skipped_trace_files:
            print(f"Skipped {len(skipped_trace_files)} large trace file(s) (> {large_file_threshold_mb} MB):")
            for name, size in skipped_trace_files:
                if isinstance(size, (int, float)):
                    print(f"  - {name} ({size:.2f} MB)")
                else:
                    print(f"  - {name} (size unknown)")
        
        if not include_trace:
            # Also exclude small trace files if include_trace is False
            files = [f for f in files if not (
                f.name == "mlperf_log_trace.json" or
                (f.name.startswith("trace_") and f.suffix == ".json")
            )]
            print(f"Excluding all trace files: {include_trace}")
        
        if not files:
            print(f"ERROR: No files to upload in {folder_path}")
            print(f"  Total files found: {len(all_files)}")
            if all_files:
                print(f"  Files found: {[f.name for f in all_files]}")
            return
        
        trace_status = "including" if include_trace else "excluding trace files"
        excluded_count = len(all_files) - len(files)
        if excluded_count > 0:
            print(f"Found {len(files)} files to upload ({trace_status}, {excluded_count} trace file(s) excluded)...")
        else:
            print(f"Found {len(files)} files to upload...")
        
        # Upload files with retry logic and error handling
        uploaded_count = 0
        failed_files = []
        
        print(f"\nStarting file uploads...")
        print(f"Files to upload: {[f.name for f in files]}")
        
        for file_path in files:
            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  Uploading {file_path.name} ({file_size_mb:.2f} MB)...", end=" ", flush=True)
                
                # Warn about large files
                if file_size_mb > 100:
                    print(f"\n    Warning: Large file ({file_size_mb:.2f} MB), upload may take time...", flush=True)
                
                max_retries = 3
                retry_delay = 2  # seconds
                
                for attempt in range(max_retries):
                    try:
                        mlflow.log_artifact(str(file_path))
                        print("✓", flush=True)
                        uploaded_count += 1
                        break
                    except (ImportError, ModuleNotFoundError) as e:
                        error_msg = str(e)
                        error_type = type(e).__name__
                        print(f"✗ FAILED", flush=True)
                        print(f"    Error type: {error_type}")
                        print(f"    Full error message: {error_msg}")
                        if 'boto3' in error_msg.lower():
                            print(f"    This appears to be a boto3 import error")
                            if not boto3_available:
                                print(f"    Install with: pip install boto3")
                            else:
                                print(f"    WARNING: boto3 check passed at startup but import failed here")
                                print(f"    This may indicate a Python environment mismatch")
                        failed_files.append((file_path.name, f"{error_type}: {error_msg}"))
                        break
                    except MlflowException as e:
                        error_msg = str(e)
                        # Check if the error is actually about boto3
                        if 'boto3' in error_msg.lower() or 'No module named' in error_msg.lower():
                            print(f"✗ FAILED", flush=True)
                            print(f"    ERROR: Missing boto3 module required by MLflow")
                            print(f"    Install with: pip install boto3")
                            print(f"    Full error: {error_msg[:300]}")
                            failed_files.append((file_path.name, "Missing boto3 dependency"))
                            break
                        elif attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                            print(f"\n    Retry {attempt + 1}/{max_retries} after {wait_time}s...", end=" ", flush=True)
                            time.sleep(wait_time)
                        else:
                            print(f"✗ FAILED", flush=True)
                            print(f"    MLflow error: {error_msg[:300]}")
                            failed_files.append((file_path.name, error_msg))
                    except Exception as e:
                        print(f"✗ FAILED", flush=True)
                        print(f"    Unexpected error: {str(e)}")
                        failed_files.append((file_path.name, str(e)))
                        break
            except Exception as e:
                print(f"✗ FAILED to process file", flush=True)
                print(f"    Error accessing file: {str(e)}")
                failed_files.append((file_path.name, f"File access error: {str(e)}"))
        
        # Summary
        print(f"\nUpload summary:")
        print(f"  Successfully uploaded: {uploaded_count}/{len(files)} files")
        if failed_files:
            print(f"  Failed uploads: {len(failed_files)}")
            for file_name, error in failed_files:
                print(f"    - {file_name}: {error[:100]}...")
        
        print(f"\nSuccessfully uploaded results to MLflow!")
        print(f"MLflow UI: {mlflow_uri}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload MLPerf results folder to MLflow server"
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing results to upload",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow server URI (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="qwen3vl_benchmark",
        help="MLflow experiment name (default: qwen3vl_benchmark)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (default: folder name)",
    )
    parser.add_argument(
        "--exclude-trace",
        dest="include_trace",
        action="store_false",
        help="Exclude trace files from upload (default: trace files are included)",
    )
    parser.add_argument(
        "--exclude-trace-pid",
        dest="exclude_trace_pid",
        action="store_true",
        help="Exclude trace_pid*.json files from upload (default: trace_pid files are included)",
    )
    
    args = parser.parse_args()
    
    # Default to True if exclude-trace was not specified
    # When using action="store_false", the attribute is only set if the flag is provided
    if not hasattr(args, 'include_trace') or getattr(args, 'include_trace', None) is None:
        args.include_trace = True
    
    # Default to False if exclude-trace-pid was not specified
    if not hasattr(args, 'exclude_trace_pid'):
        args.exclude_trace_pid = False
    
    upload_to_mlflow(
        folder_path=args.folder,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment,
        run_name=args.run_name,
        include_trace=args.include_trace,
        exclude_trace_pid=args.exclude_trace_pid,
    )


if __name__ == "__main__":
    main()
