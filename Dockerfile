FROM ubuntu:22.04

# Do not allow interactive terminal prompts
ARG DEBIAN_FRONTEND=noninteractive

# Set necessary environment variables
ENV PIP_ROOT_USER_ACTION=ignore
ENV G_MESSAGES_DEBUG=all
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
ENV GI_TYPELIB_PATH=/usr/local/lib/girepository-1.0

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    vim \
    cmake \
    libglib2.0-0 \
    libgtk+2.0-dev \
    libgirepository1.0-dev \
    ninja-build \
    libzmq5-dev \
    libjson-glib-dev \
    iputils-ping \
    iproute2 \
    netcat-traditional

# Install Python 3.9 (Encountered issues with uca-net and Python 3.10)
RUN apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.9 \
    python3-pip \
    python3-setuptools \
    python3-wheel

# Install necessary Python packages
RUN pip install --upgrade pip && pip install \
    PyGObject==3.38.0 \
    numpy \
    pytango==9.5.0 \
    scikit-image \
    imageio

WORKDIR /home

# Install libuca and uca-net binaries
RUN git clone https://github.com/ufo-kit/libuca.git && \
    # For uca-net repository it is temporarily required to checkout to uca_net_grab_send branch
    # until the same is merged to base branch.
    git clone -b uca_net_camera_grab_send https://github.com/ufo-kit/uca-net.git

WORKDIR /home/libuca

RUN mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

WORKDIR /home/uca-net

RUN mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

# Install concert from current branch
WORKDIR /home/concert

COPY . .

RUN pip install -e .

# NOTE: Concert session location inside the container: /root/.local/share/concert/. This
# contatiner location would be mapped to test directory.