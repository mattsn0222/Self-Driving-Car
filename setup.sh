#!/bin/bash
# Script to setup environment

cd ..
# Docker Setup
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo docker run hello-world
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world

pip install gevent gevent-websocket python-socketio attrdict
sudo apt-get install python-numpy python-scipy
pip install tensorflow=="1.3.0rc0"

cd Self-Driving-Car
