# Definitions
project_id = ucsd-medicine-cast
identifier = fit_dn_mt_model
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
		--platform linux/amd64 \
		-t $(name):$(tag) .
# --no-cache 
	docker tag $(name):$(tag) $(name):latest

# Push
push:
	docker push $(name):$(tag)
	docker push $(name):latest