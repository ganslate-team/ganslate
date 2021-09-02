FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

RUN apt-get -qq update
# libsm6 and libxext6 are needed for cv2
RUN apt-get update && apt-get install -y libxext6 libglib2.0-0 libsm6 build-essential sudo \
    libgl1-mesa-glx git wget rsync tmux nano dcmtk fftw3-dev liblapacke-dev libpng-dev libopenblas-dev jq && \
  rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos '' ganslate_user
USER ganslate_user

# Setup apex for mixed precision
WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/apex \
 && cd apex \
 && pip install -v --disable-pip-version-check --no-cache-dir ./ \
 && cd ..

USER root
RUN mkdir /data && chmod 777 /data
USER ganslate_user

WORKDIR /home/ganslate_user

# Install the ganslate package #TODO: Replace with final package link
RUN pip install --no-warn-script-location -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple ganslate==0.1.4

# Script are installed in ~/.local/bin, add it to PATH
ENV PATH "~/.local/bin:$PATH"

ENTRYPOINT /bin/bash
