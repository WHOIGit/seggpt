# Ubuntu 20.04; Python 3.8; CUDA 11.2.0
FROM nvcr.io/nvidia/pytorch:21.02-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    git \
    vim \
    libgl1 \
    libglib2.0-0 \
    libxdamage1

COPY ./ /main

WORKDIR /main

RUN pip install -r requirements.txt && \
    pip install git+https://github.com/facebookresearch/detectron2.git --no-build-isolation
