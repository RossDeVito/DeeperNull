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

# Copy in plink2 binary from local directory
COPY plink2 /usr/local/bin/plink2

# Copy in PRSice2 linux binary from local directory
COPY PRSice_linux /usr/local/bin/PRSice_linux
RUN chmod +x /usr/local/bin/PRSice_linux

# Install Python3 and pandas + scikit-learn
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-distutils \
    python3-dev \
    python3-pip python3-setuptools && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir pandas scikit-learn

# Invalidate cache beyond this point with a cheeky work around
# https://stackoverflow.com/questions/35134713/disable-cache-for-specific-run-commands
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache

# Copy in fit_wrapper.py from local directory
COPY fit_wrapper.py /home/fit_wrapper.py
