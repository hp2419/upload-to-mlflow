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


def parse_vllm_stdout(stdout_file: Path) -> dict:
    """Parse vLLM stdout file and extract parameters for MLflow tags.
    
    Args:
        stdout_file: Path to mlperf_log_vllm-stdout.txt
        
    Returns:
        Dictionary of tags extracted from vLLM parameters
    """
    tags = {}
    
    if not stdout_file.exists():
        print(f"Warning: vLLM stdout file not found: {stdout_file}")
        return tags
    
    content = stdout_file.read_text()
    
    # Find the "non-default args" line which contains a Python dict
    # The dict might span multiple lines, so we need to find the start and parse until the closing brace
    non_default_match = re.search(r"non-default args:\s*(\{)", content)
    if not non_default_match:
        print(f"Warning: Could not find 'non-default args' in {stdout_file}")
        return tags
    
    try:
        # Find the start position of the dictionary
        start_pos = non_default_match.end(1) - 1  # Position of the opening brace
        # Parse the dictionary by finding matching braces
        brace_count = 0
        end_pos = start_pos
        for i, char in enumerate(content[start_pos:], start=start_pos):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        
        if brace_count != 0:
            raise ValueError("Unmatched braces in dictionary")
        
        # Extract the dictionary string
        args_str = content[start_pos:end_pos]
        # Clean up the string - remove ANSI escape codes
        args_str = re.sub(r'\x1b\[[0-9;]*m', '', args_str)
        args_str = args_str.strip()
        
        # Try to parse as Python dict
        args_dict = ast.literal_eval(args_str)
        
        # Extract key parameters and create tags
        # Parallelism settings
        if 'tensor_parallel_size' in args_dict:
            tags['vllm_tp'] = str(args_dict['tensor_parallel_size'])
        if 'data_parallel_size' in args_dict:
            tags['vllm_dp'] = str(args_dict['data_parallel_size'])
        if 'pipeline_parallel_size' in args_dict:
            tags['vllm_pp'] = str(args_dict['pipeline_parallel_size'])
        if 'enable_expert_parallel' in args_dict:
            tags['vllm_ep'] = 'enabled' if args_dict['enable_expert_parallel'] else 'disabled'
        
        # Model settings
        if 'max_model_len' in args_dict:
            tags['vllm_max_model_len'] = str(args_dict['max_model_len'])
        if 'quantization' in args_dict:
            tags['vllm_quantization'] = str(args_dict['quantization'])
        if 'dtype' in args_dict:
            # dtype might be in the engine config, try to extract from there
            dtype_match = re.search(r"dtype=([\w.]+)", content)
            if dtype_match:
                tags['vllm_dtype'] = dtype_match.group(1)
        
        # Memory and performance settings
        if 'gpu_memory_utilization' in args_dict:
            tags['vllm_gpu_mem_util'] = str(args_dict['gpu_memory_utilization'])
        if 'max_num_batched_tokens' in args_dict:
            tags['vllm_max_batched_tokens'] = str(args_dict['max_num_batched_tokens'])
        
        # Scheduling and optimization
        if 'async_scheduling' in args_dict:
            tags['vllm_async_scheduling'] = 'enabled' if args_dict['async_scheduling'] else 'disabled'
        if 'enable_prefix_caching' in args_dict:
            tags['vllm_prefix_caching'] = 'enabled' if args_dict['enable_prefix_caching'] else 'disabled'
        if 'enable_chunked_prefill' in args_dict:
            # Try to find from log message
            chunked_match = re.search(r"Chunked prefill is (enabled|disabled)", content)
            if chunked_match:
                tags['vllm_chunked_prefill'] = chunked_match.group(1)
        
        # Expert parallel specific
        if 'all2all_backend' in args_dict:
            tags['vllm_all2all_backend'] = str(args_dict['all2all_backend'])
        if 'enable_eplb' in args_dict:
            tags['vllm_eplb'] = 'enabled' if args_dict['enable_eplb'] else 'disabled'
        
        # Compilation and optimization
        if 'enforce_eager' in args_dict:
            tags['vllm_enforce_eager'] = 'enabled' if args_dict['enforce_eager'] else 'disabled'
        
        # CUDA graph settings
        if 'cudagraph_capture_sizes' in args_dict:
            tags['vllm_cudagraph'] = 'enabled' if args_dict.get('cudagraph_capture_sizes') else 'disabled'
        if 'max_cudagraph_capture_size' in args_dict:
            tags['vllm_cudagraph_size'] = str(args_dict['max_cudagraph_capture_size'])
        
        # Model path/revision
        if 'revision' in args_dict:
            tags['vllm_model_revision'] = str(args_dict['revision'])
        
        # vLLM version
        version_match = re.search(r"vLLM API server version ([\d.]+)", content)
        if version_match:
            tags['vllm_version'] = version_match.group(1)
        
        # Architecture
        arch_match = re.search(r"Resolved architecture:\s*(\w+)", content)
        if arch_match:
            tags['vllm_architecture'] = arch_match.group(1)
            
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Could not parse vLLM args dictionary: {e}")
        # Fallback: try to extract key parameters with regex
        if 'tensor_parallel_size' in content:
            tp_match = re.search(r"'tensor_parallel_size':\s*(\d+)", content)
            if tp_match:
                tags['vllm_tp'] = tp_match.group(1)
        if 'data_parallel_size' in content:
            dp_match = re.search(r"'data_parallel_size':\s*(\d+)", content)
            if dp_match:
                tags['vllm_dp'] = dp_match.group(1)
        if 'enable_expert_parallel' in content:
            ep_match = re.search(r"'enable_expert_parallel':\s*(True|False)", content)
            if ep_match:
                tags['vllm_ep'] = 'enabled' if ep_match.group(1) == 'True' else 'disabled'
    
    return tags


def upload_to_mlflow(
    folder_path: str,
    mlflow_uri: str = "http://localhost:5000",
    experiment_name: str = "qwen3vl_benchmark",
    run_name: Optional[str] = None,
    include_trace: bool = True,
    exclude_trace_pid: bool = False,
) -> None:
    """Upload a folder to MLflow server as artifacts.
    
    Args:
        folder_path: Path to the folder containing results to upload
        mlflow_uri: MLflow server URI
        experiment_name: Name of the MLflow experiment
        run_name: Name for the MLflow run (defaults to folder name)
        include_trace: Whether to include trace files (default: True)
        exclude_trace_pid: Whether to exclude trace_pid*.json files (default: False)
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Set MLflow tracking URI
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
    
    # Use folder name as run name if not provided
    if run_name is None:
        run_name = folder.name
    
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
        
        # Parse and log tags from vLLM stdout
        stdout_file = folder / "mlperf_log_vllm-stdout.txt"
        if stdout_file.exists():
            print("Parsing vLLM parameters from mlperf_log_vllm-stdout.txt...")
            tags = parse_vllm_stdout(stdout_file)
            
            if tags:
                print(f"Logging {len(tags)} tags from vLLM parameters...")
                mlflow.set_tags(tags)
                for tag_name, value in tags.items():
                    print(f"  - {tag_name}: {value}")
        else:
            print(f"Warning: vLLM stdout file not found: {stdout_file}")
        
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
