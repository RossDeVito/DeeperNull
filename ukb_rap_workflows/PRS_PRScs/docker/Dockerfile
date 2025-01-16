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
    libpng-dev \
	gfortran \ 
    libblas-dev \
    liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy in plink2 binary from local directory
COPY plink2 /usr/local/bin/plink2

# Install Python3 and pandas
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-distutils \
    python3-dev \
    python3-pip \
	python3-setuptools && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir pandas

# Install PRS-CS
RUN git clone https://github.com/getian107/PRScs.git /opt/PRScs && \
	pip3 install --no-cache-dir numpy scipy h5py

# Set the environment variable
ENV PRSCS_DIR=/opt/PRScs

# Invalidate cache beyond this point
# https://stackoverflow.com/questions/35134713/disable-cache-for-specific-run-commands
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache

# Copy in ../add_null_to_covar.py
COPY prepro_covar.py /usr/local/prepro_covar.py