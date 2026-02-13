# Setting Up MLflow Upload Script on a New Host

This guide explains how to replicate the MLflow upload environment from the current system to a new host.

## Current System Configuration

Based on the current setup:
- **Python**: 3.9.21
- **MLflow**: 3.1.4
- **boto3**: 1.42.31
- **Script**: `upload_to_mlflow.py`

## Step-by-Step Setup Instructions

### 1. Install Python (if not already installed)

```bash
# Check Python version
python3 --version

# If Python 3.8+ is not installed, install it:
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install python3 python3-pip

# CentOS/RHEL:
sudo yum install python3 python3-pip
```

### 2. Create a Virtual Environment (Recommended)

#### Option A: Using venv (Python built-in)

```bash
# Create virtual environment
python3 -m venv mlflow-env

# Activate it
source mlflow-env/bin/activate

# Verify activation (prompt should show (mlflow-env))
which python
```

#### Option B: Using conda (if available)

```bash
# Create conda environment
conda create -n mlflow-test python=3.9
conda activate mlflow-test
```

### 3. Install Dependencies

```bash
# Make sure virtual environment is activated
# Install MLflow and boto3
pip install mlflow==3.1.4 boto3==1.42.31

# Or install latest versions
pip install mlflow boto3

# Verify installation
pip list | grep -E "(mlflow|boto3)"
```

### 4. Copy the Upload Script

```bash
# Create directory for scripts
mkdir -p ~/scripts/upload-to-mlflow

# Copy the script from source system
# Option 1: If you have access to the source system
scp user@source-host:/mnt/data/hpothina/upload-to-mlflow/upload_to_mlflow.py ~/scripts/upload-to-mlflow/

# Option 2: If copying manually, ensure the script is executable
chmod +x ~/scripts/upload-to-mlflow/upload_to_mlflow.py
```

### 5. Configure AWS Credentials (if using S3 for artifacts)

MLflow requires boto3 for artifact storage, which needs AWS credentials.

#### Option A: Environment Variables

```bash
# Add to ~/.bashrc or ~/.bash_profile for persistence
export AWS_ACCESS_KEY_ID="your_access_key_id"
export AWS_SECRET_ACCESS_KEY="your_secret_access_key"

# Or set temporarily for current session
export AWS_ACCESS_KEY_ID="your_access_key_id"
export AWS_SECRET_ACCESS_KEY="your_secret_access_key"
```

#### Option B: AWS Credentials File

```bash
# Create AWS credentials directory
mkdir -p ~/.aws

# Create credentials file
cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = your_access_key_id
aws_secret_access_key = your_secret_access_key
EOF

# Set proper permissions
chmod 600 ~/.aws/credentials
```

### 6. Set MLflow Tracking URI (Optional)

If you want to set a default MLflow server:

```bash
# Add to ~/.bashrc or ~/.bash_profile
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"

# Or set temporarily
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
```

### 7. Test the Setup

```bash
# Activate virtual environment (if using venv)
source mlflow-env/bin/activate

# Or activate conda environment (if using conda)
# conda activate mlflow-test

# Test Python imports
python3 -c "import mlflow; import boto3; print('✓ Dependencies installed correctly')"

# Test the script (dry run - will show help)
python3 ~/scripts/upload-to-mlflow/upload_to_mlflow.py --help
```

### 8. Create a Convenience Script (Optional)

Create a wrapper script for easier usage:

```bash
cat > ~/scripts/upload-to-mlflow/upload.sh << 'EOF'
#!/bin/bash
# Wrapper script for MLflow upload

# Activate virtual environment (adjust path as needed)
source ~/mlflow-env/bin/activate

# Or if using conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate mlflow-test

# Run the upload script with all arguments
python3 ~/scripts/upload-to-mlflow/upload_to_mlflow.py "$@"
EOF

chmod +x ~/scripts/upload-to-mlflow/upload.sh
```

### 9. Usage Example

```bash
# Activate environment
source mlflow-env/bin/activate  # or: conda activate mlflow-test

# Upload a single run directory
python3 ~/scripts/upload-to-mlflow/upload_to_mlflow.py \
    /path/to/results/run1 \
    --mlflow-uri http://150.239.115.202:5000 \
    --experiment VLM-qwen3vl \
    --run-name MY_RUN_NAME \
    --exclude-trace-pid

# Upload entire sweep directory (recursively finds all runs)
python3 ~/scripts/upload-to-mlflow/upload_to_mlflow.py \
    /path/to/sweep_h200_rc2_submitted \
    --mlflow-uri http://150.239.115.202:5000 \
    --experiment VLM-qwen3vl \
    --run-name H200_RC2_SUBMITTED_RHFP8 \
    --exclude-trace-pid
```

## Quick Setup Script

Here's a complete setup script you can run on the new host:

```bash
#!/bin/bash
# setup_mlflow_upload.sh - Complete setup script for MLflow upload environment

set -e

echo "Setting up MLflow upload environment..."

# 1. Create virtual environment
echo "Creating virtual environment..."
python3 -m venv mlflow-env
source mlflow-env/bin/activate

# 2. Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install mlflow==3.1.4 boto3==1.42.31

# 3. Create scripts directory
echo "Creating scripts directory..."
mkdir -p ~/scripts/upload-to-mlflow

# 4. Note about copying script
echo ""
echo "✓ Virtual environment created and dependencies installed"
echo ""
echo "Next steps:"
echo "1. Copy upload_to_mlflow.py to ~/scripts/upload-to-mlflow/"
echo "2. Configure AWS credentials (if needed):"
echo "   export AWS_ACCESS_KEY_ID='your_key'"
echo "   export AWS_SECRET_ACCESS_KEY='your_secret'"
echo "3. Activate environment: source mlflow-env/bin/activate"
echo "4. Run: python3 ~/scripts/upload-to-mlflow/upload_to_mlflow.py --help"
```

Save this as `setup_mlflow_upload.sh` and run:
```bash
chmod +x setup_mlflow_upload.sh
./setup_mlflow_upload.sh
```

## Verification Checklist

After setup, verify everything works:

- [ ] Python 3.8+ is installed
- [ ] Virtual environment is created and activated
- [ ] MLflow is installed (`pip list | grep mlflow`)
- [ ] boto3 is installed (`pip list | grep boto3`)
- [ ] Script is copied and executable
- [ ] AWS credentials are configured (if needed)
- [ ] MLflow server is accessible (test with `curl http://mlflow-server:5000/health`)
- [ ] Script runs without errors (`python3 upload_to_mlflow.py --help`)

## Troubleshooting

### Issue: "No module named 'mlflow'"
**Solution**: Make sure virtual environment is activated:
```bash
source mlflow-env/bin/activate
which python  # Should show path to venv python
```

### Issue: "Missing required dependency 'boto3'"
**Solution**: Install boto3:
```bash
pip install boto3
```

### Issue: "Connection refused" to MLflow server
**Solution**: 
1. Verify MLflow server is running
2. Check network connectivity: `curl http://mlflow-server:5000/health`
3. Verify firewall allows connection to port 5000

### Issue: AWS credentials not found
**Solution**: Set AWS credentials as shown in step 5 above

## Differences from Current System

If you want to match the exact current system:
- Python 3.9.21 (but 3.8+ should work)
- MLflow 3.1.4 (latest should also work)
- boto3 1.42.31 (latest should also work)

You can install exact versions:
```bash
pip install mlflow==3.1.4 boto3==1.42.31
```

## Persistent Setup (Add to .bashrc)

To make the environment activation automatic, add to `~/.bashrc`:

```bash
# MLflow upload environment
alias activate-mlflow='source ~/mlflow-env/bin/activate'
# Or for conda:
# alias activate-mlflow='conda activate mlflow-test'

# Set default MLflow URI (optional)
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
```

Then reload:
```bash
source ~/.bashrc
```

## Summary

The setup requires:
1. Python 3.8+ with pip
2. Virtual environment (venv or conda)
3. MLflow and boto3 packages
4. The upload_to_mlflow.py script
5. AWS credentials (if using S3)
6. Access to MLflow tracking server

Once set up, you can upload MLPerf results to MLflow from any directory containing run results.
