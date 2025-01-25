# docker build -t deepernull/fit_dn_model:latest -f Dockerfile .
# docker run -it --rm gcr.io/ucsd-medicine-cast/fit_dn_mt_model:latest /bin/bash
# docker run -it --rm -v "$(pwd):/mnt/loc" gcr.io/ucsd-medicine-cast/fit_dn_model:latest /bin/bash

# Options:
ARG UBUNTU_VERSION=20.04
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
	torch==1.8.1 \
	torchmetrics \
	torchvision \
	cvxpy \
	tqdm \
	xgboost

# Install LibMTL
RUN git clone https://github.com/median-research-group/LibMTL.git && \
	cd LibMTL && \
	pip3 install . --no-deps

# # Install DeeperNull
# RUN git clone https://github.com/RossDeVito/DeeperNull.git && \
# 	cd DeeperNull && \
# 	pip3 install . --no-deps
