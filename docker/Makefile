.PHONY: build setup run stop

# docker
IMAGE_NAME=how-to-optimize-gemm
CONTAINER_NAME=how-to-optimize-gemm
PACKAGE_VERSION=0.0.3
DOCKER_FILE=Dockerfile

# for release, there will be name space
NAMESPACE?=sekidocker.rnd.ki.sw.ericsson.se/proj-b5g-testbed/dev/

# ARG for Dockerfile
VCS_REF=$(PACKAGE_VERSION)
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

# SET dir

# build docker image
build:
	docker build --rm -f $(DOCKER_FILE) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		--build-arg VCS_REF=$(VCS_REF) \
		-t $(NAMESPACE)$(IMAGE_NAME):latest -t $(NAMESPACE)$(IMAGE_NAME):$(PACKAGE_VERSION) .

# run the image
HOST_NAME=$(shell hostname | xargs nslookup | grep Name | awk '{print $2}')
CURRENT_DIR=$(shell pwd)

SRC_PATH?=$(CURRENT_DIR)/../src
DATA_PATH?=$(CURRENT_DIR)/../data
BUILD_PATH?=$(CURRENT_DIR)/../build
NOTEBOOK_PATH?=$(CURRENT_DIR)/../notebooks
FIGURE_PATH?=$(CURRENT_DIR)/../figures

run:
	docker run -itd -v $(SRC_PATH):/root/src -v $(DATA_PATH):/root/data -v $(BUILD_PATH):/root/build -v $(NOTEBOOK_PATH):/root/notebooks -v $(FIGURE_PATH):/root/figures --rm --network=host --name $(CONTAINER_NAME) $(NAMESPACE)$(IMAGE_NAME) bash

stop:
	docker stop $(CONTAINER_NAME)
