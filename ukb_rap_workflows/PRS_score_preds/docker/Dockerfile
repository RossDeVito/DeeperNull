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

# Install Python packages
RUN pip3 install --no-cache-dir \
	matplotlib \
	numpy \
	pandas \
	scikit-learn \
    scikit-posthocs \
	scipy \
	seaborn \
    tqdm

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache

# Copy in score_preds.py from local directory
COPY score_preds.py /home/score_preds.py
COPY score_preds_bootstrap.py /home/score_preds_bootstrap.py