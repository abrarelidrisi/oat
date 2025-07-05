#!/bin/bash
set -e  

echo "Step 1: Cleaning up old installations..."
rm -rf .venv
rm -rf **/__pycache__
rm -rf *.egg-info
rm -rf build/
rm -rf dist/

echo "Step 2: Creating fresh virtual environment..."
python3.10 -m venv .venv
source .venv/bin/activate

echo "Step 3: Installing basic tools..."
pip install --upgrade pip
pip install uv

echo "Step 4: Installing compatible NumPy..."
uv pip install "numpy>=1.21.0,<2.0.0"

echo "Step 5: Installing PyTorch 2.2.2 with CUDA 12.1..."
uv pip install \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

echo "Step 6: Installing core ML dependencies..."
uv pip install \
    transformers \
    accelerate \
    datasets \
    tokenizers \
    safetensors \
    huggingface-hub

echo "Step 7: Installing build dependencies..."
uv pip install \
    packaging \
    setuptools \
    wheel \
    ninja \
    psutil

echo "Step 8: Installing flash-attn..."
echo "This might take a while..."

# First try: Pre-built wheel
echo "Trying pre-built wheel..."
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=${MAX_JOBS:-6}
UV_CACHE_DISABLED=1 \
uv pip install flash-attn==2.7.4.post1 \
       --no-build-isolation \
       --no-binary flash-attn \
       --no-cache-dir

echo "Step 9: Installing additional dependencies..."
uv pip install \
    wandb \
    deepspeed \
    einops \
    scipy \
    scikit-learn

echo "Step 10: Installing OAT project..."
if [ -f "pyproject.toml" ]; then
    echo "Found pyproject.toml - installing project..."
    uv pip install -e . --no-deps
    uv pip install \
        matplotlib \
        seaborn \
        jupyter \
        ipython \
        tqdm
else
    echo "No pyproject.toml found - please check if you're in the right directory"
    exit 1
fi

echo "Step 11: Final testing..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import numpy as np
    print(f'NumPy {np.__version__} working')
except Exception as e:
    print(f'NumPy issue: {e}')

try:
    import torch
    print(f'PyTorch {torch.__version__} working')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'PyTorch issue: {e}')

try:
    import transformers
    print(f'Transformers {transformers.__version__} working')
except Exception as e:
    print(f'Transformers issue: {e}')

try:
    import flash_attn
    print(f'Flash Attention {flash_attn.__version__} working')
except Exception as e:
    print(f'Flash Attention issue: {e}')
    print('You can still use OAT, but some operations might be slower')

try:
    import oat
    print('OAT imported successfully!')
except Exception as e:
    print(f'OAT import failed: {e}')
    print('Check if all dependencies are properly installed')
"

echo ""
echo "=== Installation Complete ==="
