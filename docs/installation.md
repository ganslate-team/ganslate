# Installation

You can install `ganslate` either through a docker setup or directly on your system. 
## Docker
*Supported operating systems: Linux, [Windows with WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)*

Dockerized setup is the easiest way to get started with the framework. If you do not have docker installed, you can follow instructions [here](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)


You can run the docker image, which will give you access to a container with all dependencies installed, using,

```console
docker run --gpus all -it surajpaib/ganslate:latest
```

This will drop down to a shell and [you can now check out the quickstart page](quickstart.md)



!!! note 
	To get your data into the docker container, you can use volume mounts. The docker container [mounts volumes from the host system](https://docs.docker.com/storage/volumes/) to allow easier persistence of data. This can be done as `docker run --gpus all --volume=<data_dir>:/data -it ganslate:latest`. `<data_dir>` must be replaced with the full path of a directory where your data is located, this will then be mounted on the `/data` path within the docker

## Local

!!! note
	It is recommended to use to setup a conda environment to install pytorch dependencies. You can do this by 
	[installing conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) first, then followed by `conda create env -n ganslate_env python pytorch -c pytorch`.

You can install the ganslate package along with its dependencies using
```console
pip install ganslate
```

The `ganslate` package is now installed. [You can now check out the quickstart page](quickstart.md)
