#!/bin/bash
# setup_mlflow_upload.sh - Complete setup script for MLflow upload environment
# This script replicates the MLflow upload setup on a new host

set -e

echo "=========================================="
echo "MLflow Upload Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed"
    echo "Please install Python 3.8+ first"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "   ✓ Python $PYTHON_VERSION found"

# Check if virtual environment should be created
USE_VENV=true
if [ -d "mlflow-env" ]; then
    echo ""
    read -p "Virtual environment 'mlflow-env' already exists. Use it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        USE_VENV=false
    fi
fi

# Create virtual environment
if [ "$USE_VENV" = true ]; then
    echo ""
    echo "2. Creating virtual environment..."
    python3 -m venv mlflow-env
    echo "   ✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "3. Activating virtual environment..."
source mlflow-env/bin/activate
echo "   ✓ Virtual environment activated"
echo "   Python: $(which python3)"
echo "   Python version: $(python3 --version)"

# Upgrade pip
echo ""
echo "4. Upgrading pip..."
pip install --upgrade pip --quiet
echo "   ✓ pip upgraded"

# Install dependencies
echo ""
echo "5. Installing dependencies..."
echo "   Installing mlflow==3.1.4..."
pip install mlflow==3.1.4 --quiet
echo "   Installing boto3==1.42.31..."
pip install boto3==1.42.31 --quiet
echo "   ✓ Dependencies installed"

# Verify installation
echo ""
echo "6. Verifying installation..."
python3 -c "import mlflow; import boto3; print('   ✓ MLflow version:', mlflow.__version__); print('   ✓ boto3 version:', boto3.__version__)" 2>/dev/null || {
    echo "   ✗ Failed to import mlflow or boto3"
    exit 1
}

# Create scripts directory
echo ""
echo "7. Setting up scripts directory..."
mkdir -p ~/scripts/upload-to-mlflow
echo "   ✓ Directory created: ~/scripts/upload-to-mlflow"

# Check if script exists
SCRIPT_PATH="$(dirname "$0")/upload_to_mlflow.py"
if [ -f "$SCRIPT_PATH" ]; then
    echo ""
    echo "8. Copying upload script..."
    cp "$SCRIPT_PATH" ~/scripts/upload-to-mlflow/
    chmod +x ~/scripts/upload-to-mlflow/upload_to_mlflow.py
    echo "   ✓ Script copied to ~/scripts/upload-to-mlflow/"
else
    echo ""
    echo "8. ⚠ Upload script not found in current directory"
    echo "   Please copy upload_to_mlflow.py to ~/scripts/upload-to-mlflow/"
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   source mlflow-env/bin/activate"
echo ""
echo "2. Configure AWS credentials (if using S3):"
echo "   export AWS_ACCESS_KEY_ID='your_key'"
echo "   export AWS_SECRET_ACCESS_KEY='your_secret'"
echo "   # Or create ~/.aws/credentials file"
echo ""
echo "3. Test the script:"
echo "   python3 ~/scripts/upload-to-mlflow/upload_to_mlflow.py --help"
echo ""
echo "4. Upload results:"
echo "   python3 ~/scripts/upload-to-mlflow/upload_to_mlflow.py \\"
echo "       /path/to/results \\"
echo "       --mlflow-uri http://your-mlflow-server:5000 \\"
echo "       --experiment VLM-qwen3vl"
echo ""
echo "To make activation persistent, add to ~/.bashrc:"
echo "   alias activate-mlflow='source $(pwd)/mlflow-env/bin/activate'"
echo ""
