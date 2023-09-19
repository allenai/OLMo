# FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ARG DIST
ARG TARGET
FROM --platform=linux/amd64 nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles"

# Install base tools.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    git-lfs \
    jq \
    language-pack-en \
    make \
    man-db \
    manpages \
    manpages-dev \
    manpages-posix \
    manpages-posix-dev \
    sudo \
    unzip \
    vim \
    wget \
    fish \
    parallel \
    iputils-ping \
    htop \
    emacs \
    tmux

# Install vulkan-utils without the recommended dependencies, to avoid the mesa-vulkan-drivers
# package. Installing that package causes Vulkan to include CPUs as rendering targets, which isn't
# preferable for GPU workloads for obvious reasons. People who want to run Vulkan on a CPU should
# install `mesa-vulkan-drivers` directly. See: https://github.com/allenai/beaker/issues/3140.
RUN apt-get install -y --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libxcb1-dev \
    vulkan-utils

# This ensures the dynamic linker (or NVIDIA's container runtime, I'm not sure)
# puts the right NVIDIA things in the right place (that THOR requires).
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Install conda. We give anyone in the users group the ability to run
# conda commands and install packages in the base (default) environment.
# Things installed into the default environment won't persist, but we prefer
# convenience in this case and try to make sure the user is aware of this
# with a message that's printed when the session starts.
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh \
    && echo "32d73e1bc33fda089d7cd9ef4c1be542616bd8e437d1f77afeeaf7afdb019787 Miniconda3-py310_23.1.0-1-Linux-x86_64.sh" \
        | sha256sum --check \
    && bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

ENV PATH=/opt/miniconda3/bin:/opt/miniconda3/condabin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install a few additional utilities via pip
RUN /opt/miniconda3/bin/pip install --no-cache-dir \
    gpustat \
    jupyter \
    oocmap

# Ensure users can modify their container environment.
RUN echo '%users ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Make the base image friendlier for interactive workloads. This makes things like the man command
# work.
RUN yes | unminimize

# Install MLNX OFED user-space drivers
# See https://docs.nvidia.com/networking/pages/releaseview.action?pageId=15049785#Howto:DeployRDMAacceleratedDockercontaineroverInfiniBandfabric.-Dockerfile
ENV MOFED_VER 5.8-1.1.2.1
ENV OS_VER ubuntu20.04
ENV PLATFORM x86_64
RUN wget --quiet https://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}/MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}/mlnxofedinstall --basic --user-space-only --without-fw-update -q && \
    rm -rf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM} && \
    rm MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz

# Install Docker CLI. Version matches Beaker on-premise servers.
RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-20.10.21.tgz -o docker.tgz \
    && sudo tar xzvf docker.tgz --strip 1 -C /usr/local/bin docker/docker \
    && rm docker.tgz

# Install Beaker
ARG BEAKER_VERSION
# =1.5.58
RUN curl --silent \
    --connect-timeout 5 \
    --max-time 10 \
    --retry 5 \
    --retry-delay 0 \
    --retry-max-time 40 \
    --output beaker.tar.gz \
    "https://beaker.org/api/v3/release/cli?os=linux&arch=amd64&version=${BEAKER_VERSION}" \
    && tar -zxf beaker.tar.gz -C /usr/local/bin/ ./beaker \
    && rm beaker.tar.gz


RUN mkdir -p /workspace

COPY profile.d /workspace/profile.d
COPY requirements.txt /workspace/
COPY README.md /workspace/
RUN pip install -r /workspace/requirements.txt

COPY eb_gantry /workspace/eb_gantry
COPY efficiency_benchmark /workspace/efficiency_benchmark
COPY setup.py /workspace/

WORKDIR /workspace/
RUN pip install .

ENTRYPOINT ["bash", "-l"]
