# Dockerfile for GPU DeepNull training

# Options:
ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=11.8.0

ARG BASE_IMAGE=pytorchlightning/pytorch_lightning:base-cuda-py${PYTHON_VERSION}-torch2.0-cuda${CUDA_VERSION}

# Base image
FROM ${BASE_IMAGE}

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
	torch>=2.1 \
	torchmetrics \
	tqdm \
	xgboost