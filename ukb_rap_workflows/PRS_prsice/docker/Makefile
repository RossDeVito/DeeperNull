# docker run -it --rm gcr.io/ucsd-medicine-cast/geonull_prs_prsice2:latest /bin/bash

# Definitions
project_id = ucsd-medicine-cast
identifier = nonlin_prs_prs_prsice2
version = 0.0.1

# GCR prefix
gcr_prefix = gcr.io/$(project_id)

# Image name and tag
name = $(gcr_prefix)/$(identifier)
tag = $(version)

# Build
build:
	cp ../fit_wrapper.py .
	docker build \
		--progress=plain \
		--platform linux/amd64 \
		-t $(name):$(tag) .
	docker tag $(name):$(tag) $(name):latest

# Push
push:
	docker push $(name):$(tag)
	docker push $(name):latest