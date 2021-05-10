# Installation

## Docker
*Supported operating systems: Linux, [Windows with WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)*

Dockerized setup is the easiest way to get started with the framework. If you do not have docker installed, you can follow instructions [here](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)


Here is a step-by-step walkthrough of how to install the framework with docker,

1. The docker image can be built using,

```console
cd docker/
docker build -t ganslate:latest .
```

2.  Once the docker image is built, it can be run using,

```console
docker run --gpus all -it \
	--shm-size=24gb --volume=<data_dir>:/data --volume=<code_dir>:/code \
	--name=ganslate ganslate:latest /bin/bash
```

 

The docker container [mounts volumes from the host system](https://docs.docker.com/storage/volumes/) to allow easier persistence of data. 

`<data_dir>` must be replaced with the full path of a directory where your data is located. It can also point to an empty directory during setup (Data can be moved into this directory later as docker mounts the directory to the container). 

`<code_dir>` must be replaced with the path to the `ganslate` repository. 

## Conda

The framework can be installed with Conda through following the steps,

1. Create a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) from the `environment.yaml` file 

```console
conda env create -f environment.yml
```