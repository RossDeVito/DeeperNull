# Dockerfile for GPU DeepNull training

# Options:
ARG OS_VERSION=ubuntu20.04
ARG CUDA_VERSION=11.4.3

# Base image from NVIDIA
FROM nvidia/cuda:${CUDA_VERSION}-devel-${OS_VERSION}

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    libjpeg-dev \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-distutils \
    python3-dev \
	python3-pip python3-setuptools && \
	rm -rf /var/lib/apt/lists/*

# Install Python packages for DeeperNull
RUN pip3 install --no-cache-dir \
	dxpy \
	matplotlib \
	numpy \
	pandas \
	pytorch-lightning \
	scikit-learn \
	scipy \
	seaborn \
	torch \
	torchmetrics \
	tqdm \
	xgboost