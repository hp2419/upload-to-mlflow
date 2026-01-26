# MLPerf Results Upload to MLflow

This script (`upload_to_mlflow.py`) uploads MLPerf benchmark results to an MLflow tracking server for experiment tracking and comparison.

## Features

- **Automatic metric extraction**: Parses `mlperf_log_summary.txt` to extract all performance metrics (throughput, latency percentiles, etc.)
- **Parameter logging**: Extracts test parameters (QPS, sample counts, duration, etc.) and validation results
- **vLLM configuration tracking**: Automatically extracts vLLM configuration parameters (tensor parallelism, expert parallelism, GPU memory utilization, etc.) as MLflow tags
- **Artifact upload**: Uploads all result files (logs, traces, etc.) as MLflow artifacts
- **Smart trace file handling**: Automatically skips large trace files (>50MB) to avoid upload issues, with options to exclude all trace files
- **Retry logic**: Includes retry mechanism for failed uploads with exponential backoff

## Setup

### On Client Machine

1. **Install dependencies**:
   ```bash
   pip install mlflow boto3
   ```

2. **Configure AWS credentials**:
   
   Option A: Export environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
   export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"
   ```
   
   Option B: Create `~/.aws/credentials` file:
   ```ini
   [default]
   aws_access_key_id = YOUR_ACCESS_KEY_ID
   aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
   ```

3. **Set MLflow tracking URI**:
   ```bash
   export MLFLOW_TRACKING_URI="YOUR_MLFLOW_SERVER_URI"
   ```

4. **Run the script**:
   ```bash
   python upload_to_mlflow.py /path/to/results/folder
   ```

## Prerequisites

1. **MLflow server**: A running MLflow tracking server (e.g., `http://your-mlflow-server:5000`)
2. **Python dependencies**: `mlflow` and `boto3` (see Setup section above)
3. **AWS credentials**: Required for MLflow artifact storage (see Setup section above)

## Installation

1. Copy `upload_to_mlflow.py` to your system:
   ```bash
   cp upload_to_mlflow.py /path/to/your/scripts/
   chmod +x upload_to_mlflow.py
   ```

2. Follow the Setup instructions above to configure your environment.

## Usage

### Basic Usage

Upload a single results folder:
```bash
python upload_to_mlflow.py /path/to/results/folder
```

The script will use the `MLFLOW_TRACKING_URI` environment variable if set, or default to `http://localhost:5000`.

### Advanced Usage

```bash
python upload_to_mlflow.py /path/to/results/folder \
    --mlflow-uri YOUR_MLFLOW_SERVER_URI \
    --experiment qwen3vl_benchmark \
    --run-name my_custom_run_name \
    --exclude-trace \
    --exclude-trace-pid
```

### Command-Line Arguments

- `folder` (required): Path to the folder containing MLPerf results to upload
- `--mlflow-uri` (optional): MLflow server URI (default: `http://localhost:5000` or `$MLFLOW_TRACKING_URI` if set)
- `--experiment` (optional): MLflow experiment name (default: `qwen3vl_benchmark`)
- `--run-name` (optional): Custom name for the MLflow run (default: folder name)
- `--exclude-trace` (optional): Exclude all trace files from upload (default: trace files are included)
- `--exclude-trace-pid` (optional): Exclude `trace_pid*.json` files from upload (default: included)

### Example: Upload Multiple Results

To upload multiple result folders, you can use a simple loop:

```bash
for dir in /path/to/results/*/; do
    python upload_to_mlflow.py "$dir" \
        --mlflow-uri YOUR_MLFLOW_SERVER_URI \
        --experiment qwen3vl_sweep_results
done
```

## What Gets Uploaded

### Metrics (from `mlperf_log_summary.txt`)
- `samples_per_second`: Benchmark throughput
- `min_latency_ns`, `max_latency_ns`, `mean_latency_ns`: Latency statistics
- `p50_latency_ns`, `p90_latency_ns`, `p95_latency_ns`, `p97_latency_ns`, `p99_latency_ns`, `p99_9_latency_ns`: Latency percentiles
- All latency metrics are also stored in seconds (e.g., `mean_latency_s`)

### Parameters (from `mlperf_log_summary.txt`)
- Test configuration: `samples_per_query`, `target_qps`, `target_latency_ns`
- Duration settings: `min_duration_ms`, `max_duration_ms`
- Query settings: `min_query_count`, `max_query_count`, `max_async_queries`
- Random seeds: `qsl_rng_seed`, `sample_index_rng_seed`, `schedule_rng_seed`
- Validation flags: `result_valid`, `min_duration_satisfied`, `min_queries_satisfied`, `early_stopping_satisfied`
- Test metadata: `scenario`, `mode`, `sut_name`

### Tags (from `mlperf_log_vllm-stdout.txt`)
- Parallelism: `vllm_tp`, `vllm_dp`, `vllm_pp`, `vllm_ep`
- Model settings: `vllm_max_model_len`, `vllm_quantization`, `vllm_dtype`
- Memory: `vllm_gpu_mem_util`, `vllm_max_batched_tokens`
- Optimization: `vllm_async_scheduling`, `vllm_prefix_caching`, `vllm_chunked_prefill`, `vllm_cudagraph`
- Version info: `vllm_version`, `vllm_architecture`, `vllm_model_revision`

### Artifacts
All files in the results folder are uploaded as artifacts, except:
- Large trace files (>50MB) are automatically skipped
- Trace files can be excluded with `--exclude-trace` flag
- `trace_pid*.json` files can be excluded with `--exclude-trace-pid` flag

## File Structure Expected

The script expects the following files in the results folder:
- `mlperf_log_summary.txt` (required for metrics/parameters)
- `mlperf_log_vllm-stdout.txt` (required for vLLM tags)
- Other log files (optional, uploaded as artifacts)

## Troubleshooting

### Error: "Missing required dependency 'boto3'"
**Solution**: Install boto3:
```bash
pip install boto3
```

### Error: "MLflow requires boto3 for artifact storage"
**Solution**: Even for remote MLflow backends, boto3 is required. Install it:
```bash
pip install boto3
```

### Error: Connection refused to MLflow server
**Solution**: 
1. Ensure your MLflow server is running and accessible
2. Check that `MLFLOW_TRACKING_URI` is set correctly:
   ```bash
   echo $MLFLOW_TRACKING_URI
   ```
3. Verify network connectivity to the MLflow server:
   ```bash
   curl YOUR_MLFLOW_SERVER_URI
   ```

### Error: AWS credentials not found
**Solution**: 
1. Ensure AWS credentials are configured (see Setup section)
2. Verify credentials are exported:
   ```bash
   echo $AWS_ACCESS_KEY_ID
   echo $AWS_SECRET_ACCESS_KEY
   ```
3. Or verify `~/.aws/credentials` file exists and is readable

### Large trace files causing upload failures
**Solution**: Use the `--exclude-trace` flag to skip all trace files:
```bash
python upload_to_mlflow.py /path/to/results --exclude-trace
```

### Upload timeout or slow uploads
**Solution**: The script automatically skips trace files larger than 50MB. For very large result folders, consider:
1. Using `--exclude-trace` to skip all trace files
2. Using `--exclude-trace-pid` to skip trace_pid*.json files
3. Ensuring your MLflow backend has sufficient storage

## Viewing Results in MLflow

After uploading, you can view your results in the MLflow UI:

1. Open your browser and navigate to the MLflow server URI (e.g., `http://your-mlflow-server:5000`)
2. Select your experiment from the left sidebar
3. Compare runs using the table view or parallel coordinates plot
4. View detailed metrics, parameters, and artifacts for each run

## Example Workflow

1. **Setup environment** (one-time):
   ```bash
   pip install mlflow boto3
   export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
   export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"
   export MLFLOW_TRACKING_URI="YOUR_MLFLOW_SERVER_URI"
   ```

2. **Run MLPerf benchmark**:
   ```bash
   ./run_benchmark.sh
   ```

3. **Upload results to MLflow**:
   ```bash
   python upload_to_mlflow.py ./results/run_offline_performance_TP4_DP1_EPon_GM0.90 \
       --experiment qwen3vl_sweep
   ```

4. **View results in MLflow UI** at `YOUR_MLFLOW_SERVER_URI`

## Notes

- The script automatically creates the experiment if it doesn't exist
- Run names default to the folder name but can be customized
- Large trace files (>50MB) are automatically skipped to prevent upload issues
- The script includes retry logic for transient network errors
- All metrics and parameters are automatically parsed from the MLPerf summary file
- AWS credentials are required for MLflow artifact storage, even when using a remote MLflow server