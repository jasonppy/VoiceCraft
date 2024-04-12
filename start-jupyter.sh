#!/usr/bin/env bash
## Assumes you have docker installed with nvidia container container-toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.13.5/install-guide.html
# sudo apt-get install -y nvidia-container-toolkit-base || yay -Syu nvidia-container-toolkit || echo etc...
## Try to start an existing container otherwise create a new one
docker start jupyter 2> /dev/null || \
docker run -it \
    -d \
    --gpus all \
    -p 8888:8888 \
    -p 7860:7860 \
    --name jupyter \
    --user root \
    -e NB_USER="$USER" \
    -e CHOWN_HOME=yes \
    -e GRANT_SUDO=yes \
    -w "/home/${NB_USER}" \
    -v "$PWD":"/home/$USER/work" \
    voicecraft

## `docker logs jupyter` to get the URL link and token e.g.
## http://127.0.0.1:8888/lab?token=blahblahblahblabhlaabhalbhalbhal
