#!/usr/bin/env bash
set -e

IMAGE_NAME=alex-rl:latest
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
CONTAINER_NAME=alex_devenv

ENV_LOCATION=../.env

USER_ID=$(id -u)
GROUP_ID=$(id -g)
USERNAME=alex_dev

welcome=$(cat docker/welcome.txt)

RESET_CONTAINER=false
if [[ "$1" == "--reset" || "$1" == "-r" ]]; then
    RESET_CONTAINER=true
fi

if $RESET_CONTAINER; then
    echo "Resetting existing container '${CONTAINER_NAME}'..."
    docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true
fi

docker run -it --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name ${CONTAINER_NAME} \
    -p 8888:8888 \
    -p 6006:6006 \
    -p 9899:9899 \
    -v ${REPO_ROOT}:/alex \
    -w /alex \
    ${IMAGE_NAME} \
    bash -c "
        getent group ${GROUP_ID} || groupadd -g ${GROUP_ID} ${USERNAME}
        id -u ${USER_ID} >/dev/null 2>&1 || useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME}
        clear
        echo '${welcome}'
        export '${ENV_LOCATION}' && set -a && source ${ENV_LOCATION} && set +a
        exec su ${USERNAME} -c '/bin/bash'
    "
