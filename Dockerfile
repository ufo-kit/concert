FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PIP_ROOT_USER_ACTION=ignore
ENV G_MESSAGES_DEBUG=all
ENV LD_LIBRARY_PATH=/usr/local/lib

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    git \
    vim \
    cmake \
    libglib2.0-0 \
    libgtk+2.0-dev \
    gobject-introspection \
    ninja-build \
    libzmq5-dev \
    libjson-glib-dev

RUN pip install --upgrade pip && pip install \
    numpy \
    pytango==9.5.0 \
    scikit-image \
    meson

WORKDIR /home

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

WORKDIR /home/concert

COPY . .

RUN pip install -e .

# Concert session location inside the container: /root/.local/share/concert/