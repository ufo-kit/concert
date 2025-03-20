FROM ubuntu:24.04

# Set necessary environment variables
ENV TZ=DE
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
ENV G_MESSAGES_DEBUG=all
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
ENV GI_TYPELIB_PATH=/usr/local/lib/girepository-1.0

# Install build dependencies. Packages iputils-ping, iproute2, netcat-traditional and tree
# are included for the ease of troubleshooting.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    vim \
    cmake \
    libglib2.0-0 \
    libgtk+2.0-dev \
    libgirepository-1.0-dev \
    ninja-build \
    libzmq5-dev \
    libjson-glib-dev \
    iputils-ping \
    iproute2 \
    netcat-traditional \
    tree

RUN apt install -y \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-gi

# Install necessary Python packages
RUN pip install --break-system-packages \
    numpy \
    pytango \
    scikit-image \
    imageio \
    pytest

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

RUN pip install --break-system-packages -e .

# NOTE: Concert session location inside the container: /root/.local/share/concert/.