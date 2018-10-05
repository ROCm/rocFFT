# Parameters related to building hip
ARG base_image

FROM ${base_image}
LABEL maintainer="kent.knox@amd"

ARG user_uid

# Install dependent packages
# Dependencies:
# * hcc-config.cmake: pkg-config
# * tensile: python34, PyYAML
# * rocfft-test: gcc-gfortran, googletest
# * rocfft-bench: boost-devel
# * libhsakmt.so: libnuma1
RUN dnf -y update && dnf install -y \
    sudo \
    ca-certificates \
    git \
    make \
    cmake \
    clang \
    gcc-c++ \
    gcc-gfortran \
    python34 \
    PyYAML \
    libcxx-devel \
    rpm-build \
    boost-devel \
    && \
    dnf -y clean all

# docker pipeline runs containers with particular uid
# create a jenkins user with this specific uid so it can use sudo priviledges
# Grant any member of sudo group password-less sudo privileges
RUN useradd --create-home -u ${user_uid} -o -G wheel --shell /bin/bash jenkins && \
    mkdir -p /etc/sudoers.d/ && \
    echo '%wheel   ALL=(ALL) NOPASSWD:ALL' | tee /etc/sudoers.d/sudo-nopasswd

ARG ROCFFT_SRC_ROOT=/usr/local/src/rocFFT

# Clone rocfft repo
# Build client dependencies and install into /usr/local (LAPACK & GTEST)
RUN mkdir -p ${ROCFFT_SRC_ROOT} && cd ${ROCFFT_SRC_ROOT} && \
    git clone -b develop --depth=1 https://github.com/ROCmSoftwarePlatform/rocFFT . && \
    mkdir -p build/deps && cd build/deps && \
    cmake -DBUILD_BOOST=OFF ${ROCFFT_SRC_ROOT}/deps && \
    make -j $(nproc) install && \
    rm -rf ${ROCFFT_SRC_ROOT}
