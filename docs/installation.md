# Installation

You can begin the install by cloning the repo,
```
git clone https://github.com/Maastro-CDS-Imaging-Group/ganslate.git
```

Once the repo is fetched, you can either use a docker-based install or a conda-based install on your local system.

!!! note 
	You can get to the repository root level after cloning,
	```
	cd ganslate
	```


## Docker
*Supported operating systems: Linux, [Windows with WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)*

Dockerized setup is the easiest way to get started with the framework. If you do not have docker installed, you can follow instructions [here](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)


First, you can build the docker image by running the following commands at the repository root level

```console
cd docker/
docker build -t ganslate:latest .
```

Once the docker image is built, it can be run using,

```console
docker run --gpus all -it \
	--shm-size=24gb --volume=<data_dir>:/data --volume=<repo_dir>:/ganslate \
	--name=ganslate ganslate:latest /bin/bash
```

The docker container [mounts volumes from the host system](https://docs.docker.com/storage/volumes/) to allow easier persistence of data. 

`<data_dir>` must be replaced with the full path of a directory where your data is located.
`<repo_dir>` must be replaced with the path to the `ganslate` repository was cloned. 

!!! note
	`<data_dir>` can initially point to an empty directory to simplify setup. Data can be moved into this directory later as docker mounts the directory to the container. 

## Local

You can install the ganslate package along with its dependencies using
```console
python setup.py install
```

!!! note
	If you want to work on internals in the package then it is recommended to use `python setup.py develop`.
	https://setuptools.readthedocs.io/en/latest/userguide/development_mode.html

The `ganslate` package is now installed. [You can now check out the quickstart page](quickstart.md)
