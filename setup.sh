#!/bin/bash
# Script to setup environment

cd ..
pip install gevent gevent-websocket python-socketio attrdict
sudo apt-get install python-numpy python-scipy

# Docker Setup
sudo apt-get update
sudo apt-get install \
apt-transport-https \
ca-certificates \
curl \
gnupg-agent \
software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) \
stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo docker run hello-world

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world

sudo docker pull tensorflow/tensorflow:1.3.0

cd Self-Driving-Car
