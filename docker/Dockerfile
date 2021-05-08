ARG CUDA="10.2"
ARG CUDNN="7"

# TODO: conda installs its own version of cuda
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

ENV CUDA_PATH /usr/local/cuda
ENV CUDA_ROOT /usr/local/cuda/bin
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64

RUN apt-get -qq update
# libsm6 and libxext6 are needed for cv2
RUN apt-get update && apt-get install -y libxext6 libsm6 libxrender1 build-essential sudo \
    libgl1-mesa-glx git wget rsync tmux nano dcmtk fftw3-dev liblapacke-dev libpng-dev libopenblas-dev jq && \
  rm -rf /var/lib/apt/lists/*
RUN ldconfig

# Make a user
RUN adduser --disabled-password --gecos '' midagan
RUN adduser midagan sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER midagan

WORKDIR /tmp
RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH "/home/midagan/miniconda3/bin:/tmp/bart/:$PATH:$CUDA_ROOT"

# Setup python packages
RUN conda update -n base conda -yq
RUN conda install python=3.7
RUN conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing boost
RUN conda install pytorch==1.8 torchvision cudatoolkit=="10.2" -c pytorch
RUN conda install scipy pandas scikit-learn scikit-image=0.14 -yq
RUN conda install cython tqdm jupyter sqlalchemy -yq

RUN python -m pip install opencv-python simpleitk h5py -q
RUN python -m pip install tb-nightly wandb -q
RUN python -m pip install monai memcnn -q
RUN python -m pip install omegaconf loguru -q

USER root

# Create runner for input and output
RUN mkdir /data
RUN mkdir /runner && chmod 777 /runner

USER runner

RUN jupyter notebook --generate-config
ENV CONFIG_PATH "/home/runner/.jupyter/jupyter_notebook_config.py"
COPY "jupyter_notebook_config.py" ${CONFIG_PATH}

# Copy files into the docker
COPY [".", "/runner"]

WORKDIR /runner

ARG username
ARG pass
RUN git clone https://${username}:${pass}@github.com/Maastro-CDS-Imaging-Group/midaGAN.git

# Provide an open entrypoint for the docker
ENTRYPOINT $0 $@
