#!/bin/bash

# start-docker.sh

set -e

IMAGE_NAME="yuhui11/mmcl"

# create named container if arg given (--rm default)
if [ -z $1 ]; then
    CNAME="--rm"
else
    CNAME="--name $1"
    # if container already exists, confirm removal
    if [ $(docker ps -aq -f name=$1) ]; then
       read -p "Container already exists, overwrite? (y/n) " input
       if [ $input == "y" ]; then
           docker rm $1
       else
           exit
       fi
    fi
fi

# start bash
sudo docker run $CNAME --privileged \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined\
  $IMAGE_NAME mmcl -f cube2.inp -s cube2 -n 1e8 -b 0 -D TP -M G -F bin
