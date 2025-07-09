#!/bin/bash

# Update and fix packages
apt update && apt upgrade -y
apt --fix-broken install -y

# Install NVIDIA drivers
apt install -y nvidia-driver-575 || apt install -y nvidia-driver-550

# Add CUDA repository for Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update

# Install CUDA 12.4
apt install -y cuda-12-4 || apt install -y cuda-toolkit-12-4

# Set environment variables
echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
nvidia-smi

echo "Setup complete. Check output above for success."