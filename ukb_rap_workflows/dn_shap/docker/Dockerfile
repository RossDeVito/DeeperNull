# Options:
# ARG PYTHON_VERSION=3.10
ARG UBUNTU_VERSION=22.04
ARG BASE_IMAGE=ubuntu:${UBUNTU_VERSION}

# Base image
FROM ${BASE_IMAGE}

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
	torch>=2.0 \
	torchmetrics \
	tqdm \
	xgboost \
	shapiq

# List installed packages
RUN pip3 list
