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
	wget && \
    rm -rf /var/lib/apt/lists/*

# Install Python3 and pandas + scikit-learn
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-distutils \
    python3-dev \
    python3-pip python3-setuptools && \
    rm -rf /var/lib/apt/lists/*

# Install LDAK
RUN wget https://github.com/dougspeed/LDAK/raw/main/ldak6.linux \
	-O /usr/local/bin/ldak6 && \
	chmod +x /usr/local/bin/ldak6

# Set LDAK path
ENV LDAK_PATH=/usr/local/bin/ldak6
