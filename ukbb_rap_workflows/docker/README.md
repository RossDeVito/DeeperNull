# Dockers for DeeperNull training

Each directory contains:

* `Dockerfile` to build the relevant docker image
* `Makefile` to build and push the image. `make build` builds, `make push` pushes.

Dockers are pushed to our `gcr.io/ucsd-medicine-cast` repository.