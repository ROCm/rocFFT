# Parameters related to building rocfft
ARG base_image

FROM ${base_image}
MAINTAINER Kent Knox <kent.knox@amd>

ARG user_uid

# Install Packages & dependent packages
# Dependencies:
# * hcc-config.cmake: pkg-config
# * rocblas-test: libfftw3-dev, googletest
# * rocfft-bench: libboost-program-options-dev
# * rocm_agent_enumerator: python, libnuma1
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    sudo \
    build-essential \
    cmake \
    ca-certificates \
    git \
    pkg-config \
    libfftw3-dev \
    libboost-program-options-dev \
    python \
    libnuma1 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# docker pipeline runs containers with particular uid
# create a jenkins user with this specific uid so it can use sudo priviledges
# Grant any member of sudo group password-less sudo privileges
RUN useradd --create-home -u ${user_uid} -o -G sudo --shell /bin/bash jenkins && \
    mkdir -p /etc/sudoers.d/ && \
    echo '%sudo   ALL=(ALL) NOPASSWD:ALL' | tee /etc/sudoers.d/sudo-nopasswd
