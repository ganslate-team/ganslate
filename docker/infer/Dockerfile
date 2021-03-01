# ARG CUDA="10.2"
# ARG CUDNN="7"

# TODO: conda installs its own version of cuda
FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

# ENV CUDA_PATH /usr/local/cuda
# ENV CUDA_ROOT /usr/local/cuda/bin
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64

RUN apt-get -qq update
# libsm6 and libxext6 are needed for cv2
RUN apt-get update && apt-get install -y plastimatch libxext6 libsm6 libxrender1 build-essential sudo \
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

# ENV PATH "/home/midagan/miniconda3/bin:/tmp/bart/:$PATH:$CUDA_ROOT"
ENV PATH "/home/midagan/miniconda3/bin::$PATH"

# Setup python packages
RUN conda update -n base conda -yq
RUN conda install python=3.8
RUN conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing boost

# RUN conda install pytorch=1.7 torchvision cudatoolkit=${CUDA} -c pytorch
RUN conda install pytorch=1.7 torchvision cpuonly -c pytorch

RUN conda install scipy pandas scikit-learn scikit-image -yq
RUN conda install cython tqdm jupyter sqlalchemy -yq
RUN python -m pip install opencv-python simpleitk h5py -q
RUN python -m pip install tb-nightly memcnn wandb -q
RUN python -m pip install omegaconf -q


USER root
RUN mkdir /midagan && chmod 777 /midagan

USER midagan
ENV PYTHONPATH /midagan
WORKDIR /midagan

ARG username
ARG pass

# Clone repo for preprocessing
RUN git clone https://${username}:${pass}@github.com/Maastro-CDS-Imaging-Group/data-pipelines.git
RUN python -m pip install -r data-pipelines/requirements.txt

# Run actions for data-pipelines
WORKDIR data-pipelines

# Clone repo for midaGAN
WORKDIR /midagan
RUN git clone https://${username}:${pass}@github.com/Maastro-CDS-Imaging-Group/midaGAN.git
WORKDIR midaGAN
RUN git checkout origin/inference_docker
RUN pip install -r requirements.txt

# Clone repo for dicom saving of generated images
WORKDIR /midagan
RUN git clone https://${username}:${pass}@github.com/Maastro-CDS-Imaging-Group/clinical-evaluation.git
# Provide an open entrypoint for the docker
ENTRYPOINT ["/bin/bash", "/midagan/midaGAN/docker/infer/run_inference_pipeline.sh"]
# ENTRYPOINT $0 $@
