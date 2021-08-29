FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

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


RUN adduser --disabled-password --gecos '' runner
# RUN adduser runner sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
ARG USER_ID
ARG GROUP_ID
RUN usermod -u ${USER_ID} runner
RUN usermod -g ${GROUP_ID} runner
USER runner


WORKDIR /tmp
RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH "/home/runner/miniconda3/bin:$PATH:$CUDA_ROOT"

# Setup python packages
RUN conda update -n base conda -yq
RUN conda install python=3.8
RUN conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing boost
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
RUN conda install scipy pandas scikit-learn scikit-image -yq
RUN conda install cython tqdm jupyter sqlalchemy -yq

RUN git clone https://github.com/NVIDIA/apex \
 && cd apex \
 && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
 && cd ..

RUN python -m pip install opencv-python simpleitk h5py -q
RUN python -m pip install tb-nightly wandb -q
RUN python -m pip install monai memcnn -q
RUN python -m pip install omegaconf loguru -q


USER root

# Create folders for input and output
RUN mkdir /ganslate && chmod 777 /ganslate
RUN mkdir /data && chmod 777 /data
USER runner

WORKDIR /ganslate

ENTRYPOINT $0 $@
