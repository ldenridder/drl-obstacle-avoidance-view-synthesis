# Stereo View Synthesis

## Getting Started

These instructions will guide you through the process of setting up and running the project.

### Prerequisites

Start by obtaining the necessary dataset for this project

Next, install the required packages from the included conda and pip files.

### Running the Project

To start the project, run `main.py`. Please adjust the config settings as needed.

# DRL Obstacle Avoidance

To use the DRL Obstacle Avoidance a docker image is provided. 

## Getting Started
Before you proceed, you will need to download and install Unity from AvoidBench. The Unity package can be obtained [here](https://github.com/tudelft/AvoidBench/releases/download/v0.0.2/AvoidBench.zip).

Once Unity is installed, you can proceed to run the Docker container with the following commands:

```bash
docker run -it --device=/dev/dri --group-add video --volume=/tmp/.X11-unix:/tmp/.X11-unix --env="DISPLAY=$DISPLAY" --env="QT_X11_NO_MITSHM=1" --gpus 0 --name=drl-00 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics -e NVIDIA_VISIBLE_DEVICES=all lucdenridder/drl-obstacle-avoidance:latest /bin/bash
docker start drl-00
docker exec -it drl-00 bash
cd AvoidBench
source devel.setup.bash
roslaunch avoid_manage rl_py.launch
