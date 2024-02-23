# Dockerfile for GPU DeepNull training

ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.2
ARG CUDA_VERSION=12.1.0

# Pytorch Lightning base image
ARG TAG=base-cuda-py${PYTHON_VERSION}-torch${TORCH_VERSION}-cuda${CUDA_VERSION}
FROM pytorchlightning/pytorch_lightning:${TAG}

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
	torch \
	torchmetrics \
	tqdm \
	xgboost

# List installed packages
RUN pip3 list
RUN python3 -V
RUN python3 -c "import torch; print(torch.__version__, flush=True); print(torch.cuda.is_available(), flush=True)"
