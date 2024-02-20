# Dockerfile for GPU DeepNull training
# docker run -it --rm deepernull/fit_dn_model_gpu:latest /bin/bash

# Base image from NVIDIA
FROM nvcr.io/nvidia/pytorch:21.08-py3

# Install Python packages for DeeperNull
RUN pip3 install \
	dxpy \
	matplotlib \
	numpy \
	pandas \
	pytorch-lightning \
	scikit-learn \
	scipy \
	seaborn \
	torch==1.10.0a0+3fd9dcf \
	torchmetrics \
	tqdm \
	xgboost

# Check if the installation was successful
RUN python3 -c "import torch; print(torch.__version__, flush=True)" >> output.txt
RUN python3 -c "import torch; print(torch.cuda.is_available(), flush=True)" >> output.txt
COPY output.txt .

# # Check if the installation was successful
# RUN python3 -c "import torch; print(torch.__version__, flush=True)"
# RUN python3 -c "import torch; print(torch.cuda.is_available(), flush=True)"