#!/bin/bash

# start-docker.sh

set -e

IMAGE_NAME="yuhui11/mmcl"


# start bash
docker run --rm --privileged \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined\
  $IMAGE_NAME mmcl $@
