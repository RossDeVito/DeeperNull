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
    wget \
    libjpeg-dev \
    libpng-dev \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Make python also use python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy in plink2 binary from local directory
COPY plink2 /usr/local/bin/plink2

# Install cat-bgen
RUN wget http://code.enkre.net/bgen/tarball/release/bgen.tgz && \
	tar -xvzf bgen.tgz && \
	cd bgen.tgz && \
	./waf configure && \
	./waf

RUN cd bgen.tgz && \
	./build/test/unit/test_bgen && \
	./build/apps/bgenix -g example/example.16bits.bgen -list

RUN cd bgen.tgz && \
	./waf install && \
	cat-bgen -help
