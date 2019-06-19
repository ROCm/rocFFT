# Parameters related to building hip
ARG base_image

FROM ${base_image}
LABEL maintainer="akila.premachandra@amd.com"

ARG user_uid

RUN yum install -y \
    sudo \
    rock-dkms \
    centos-release-scl \
    devtoolset-7 \
    ca-certificates \
    git \
    cmake3 \
    make \
    clang \
    clang-devel \
    gcc-c++ \
    gcc-gfortran \
    gtest-devel \
    gtest \
    fftw-devel \
    pkgconfig \
    python27 \
    python34 \
    PyYAML \
    libcxx-devel \
    boost-devel\
    numactl-libs \
    rpm-build

#RUN echo 'scl enable devtoolset-7 bash' >>/etc/skel/.bashrc

RUN echo '#!/bin/bash' | tee /etc/profile.d/devtoolset7.sh && echo \
    'source scl_source enable devtoolset-7' >>/etc/profile.d/devtoolset7.sh

# docker pipeline runs containers with particular uid
# create a jenkins user with this specific uid so it can use sudo priviledges
# Grant any member of sudo group password-less sudo privileges
RUN useradd --create-home -u ${user_uid} -o -G video --shell /bin/bash jenkins && \
    echo '%video ALL=(ALL) NOPASSWD:ALL' | tee /etc/sudoers.d/sudo-nopasswd && \
    chmod 400 /etc/sudoers.d/sudo-nopasswd
    
ARG ROCFFT_SRC_ROOT=/usr/local/src/rocFFT

# Clone rocfft repo
# Build client dependencies and install into /usr/local (LAPACK & GTEST)
RUN mkdir -p ${ROCFFT_SRC_ROOT} && cd ${ROCFFT_SRC_ROOT} && \
    git clone -b develop --depth=1 https://github.com/ROCmSoftwarePlatform/rocFFT . && \
    mkdir -p build/deps && cd build/deps && \
    cmake3 -DBUILD_BOOST=OFF ${ROCFFT_SRC_ROOT}/deps && \
    make -j $(nproc) install && \
    rm -rf ${ROCFFT_SRC_ROOT}