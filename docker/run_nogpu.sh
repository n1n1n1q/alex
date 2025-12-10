#!/usr/bin/env bash
set -e

IMAGE_NAME=alex-rl:latest
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
CONTAINER_NAME=alex_devenv

ENV_LOCATION=.env

USER_ID=$(id -u)
GROUP_ID=$(id -g)
USERNAME=alex_dev

MEMORY=${1:-10g}
MEMORY_SWAP=${2:-11g}
CPU=${3:-4}

welcome=$(cat docker/welcome.txt)

RESET_CONTAINER=false
if [[ "$1" == "--reset" || "$1" == "-r" ]]; then
    RESET_CONTAINER=true
fi

if $RESET_CONTAINER; then
    echo "Resetting existing container '${CONTAINER_NAME}'..."
    docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true
fi

xhost +local:docker

docker run -it --rm \
    --cpus ${CPU} \
    --ipc=host \
    --name ${CONTAINER_NAME} \
    -p 8888:8888 \
    -p 6006:6006 \
    -p 9899:9899 \
    -v ${REPO_ROOT}:/alex \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -w /alex \
    --user root \
    --memory ${MEMORY} \
    --memory-swap ${MEMORY_SWAP} \
    ${IMAGE_NAME} \
    bash -c "
        echo '${welcome}'
        set -a && source ${ENV_LOCATION} && set +a
        /bin/bash
    "
