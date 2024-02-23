# Definitions
project_id = ucsd-medicine-cast
identifier = fit_dn_model_gpu
version = 0.0.1

# GCR prefix
gcr_prefix = gcr.io/$(project_id)

# Image name and tag
name = $(gcr_prefix)/$(identifier)
tag = $(version)

# Build
build:
	docker build \
		--progress=plain \
		--no-cache \
		--platform linux/amd64 \
		-t $(name):$(tag) .
	docker tag $(name):$(tag) $(name):latest

# Push
push:
	docker push $(name):$(tag)
	docker push $(name):latest