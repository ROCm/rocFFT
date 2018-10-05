# Parameters related to building hip
ARG base_image

FROM ${base_image}
LABEL maintainer="kent.knox@amd"

ARG user_uid

# Install dependent packages
# Dependencies:
# * hcc-config.cmake: pkg-config
# * tensile: python2.7, python-yaml
# * rocfft-test: gfortran, googletest
# * rocfft-bench: libboost-program-options-dev
# * libhsakmt.so: libnuma1
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    rock-dkms \
    sudo \
    build-essential \
    ca-certificates \
    git \
    make \
    cmake \
    libfftw3-dev \
    clang-format-3.8 \
    pkg-config \
    python2.7 \
    python-yaml \
    gfortran \
    libboost-program-options-dev \
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

ARG ROCFFT_SRC_ROOT=/usr/local/src/rocFFT

# Clone rocfft repo
# Build client dependencies and install into /usr/local (LAPACK & GTEST)
RUN mkdir -p ${ROCFFT_SRC_ROOT} && cd ${ROCFFT_SRC_ROOT} && \
    git clone -b develop --depth=1 https://github.com/ROCmSoftwarePlatform/rocFFT . && \
    mkdir -p build/deps && cd build/deps && \
    cmake -DBUILD_BOOST=OFF ${ROCFFT_SRC_ROOT}/deps && \
    make -j $(nproc) install && \
    rm -rf ${ROCFFT_SRC_ROOT}
