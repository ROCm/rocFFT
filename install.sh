#!/usr/bin/env bash
# Author: Kent Knox

# #################################################
# Pre-requisites check
# #################################################
# Exit code 0: alls well
# Exit code 1: problems with getopt
# Exit code 2: problems with supported platforms

# check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
  echo "This script uses getopt to parse arguments; try installing the util-linux package";
  exit 1
fi

# lsb-release file describes the system
if [[ ! -e "/etc/lsb-release" ]]; then
  echo "This script depends on the /etc/lsb-release file"
  exit 2
fi
source /etc/lsb-release

if [[ ${DISTRIB_ID} != Ubuntu ]]; then
  echo "This script only validated with Ubuntu"
  exit 2
fi

# #################################################
# helper functions
# #################################################
function display_help()
{
  echo "rocfft build & installation helper script"
  echo "./install [-h|--help] "
  echo "    [-h|--help] prints this help message"
  echo "    [-i|--install] install after build"
  echo "    [-d|--dependencies] install build dependencies"
  echo "    [-c|--clients] build library clients too (combines with -i & -d)"
  echo "    [--cuda] build library for cuda backend"
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
elevate_if_not_root( )
{
  local uid=$(id -u)

  if (( ${uid} )); then
    sudo $@
  else
    $@
  fi
}

# #################################################
# global variables
# #################################################
install_package=false
install_dependencies=false
build_clients=false
build_cuda=false

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,clients,dependencies,cuda --options hicd -- "$@")
else
  echo "Need a new version of getopt"
  exit 1
fi

if [[ $? -ne 0 ]]; then
  echo "getopt invocation failed; could not parse the command line";
  exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
  case "${1}" in
    -h|--help)
        display_help
        exit 0
        ;;
    -i|--install)
        install_package=true
        shift ;;
    -d|--dependencies)
        install_dependencies=true
        shift ;;
    -c|--clients)
        build_clients=true
        shift ;;
    --cuda)
        build_cuda=true
        shift ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
  esac
done

build_dir=./build
printf "\033[32mCreating project build directory in: \033[33m${build_dir}\033[0m\n"

# #################################################
# prep
# #################################################
# ensure a clean build environment
rm -rf ${build_dir}

# #################################################
# install build dependencies on request
# #################################################
if [[ "${install_dependencies}" == true ]]; then
  # dependencies needed for rocfft and clients to build
  library_dependencies_ubuntu=( "make" "cmake-curses-gui" "hip_hcc" "pkg-config" )
  if [[ "${build_cuda}" == false ]]; then
    library_dependencies_ubuntu+=( "hcc" )
  else
    # Ideally, this could be cuda-cufft-dev, but the package name has a version number in it
    library_dependencies_ubuntu+=( "cuda" )
  fi

  client_dependencies_ubuntu=( "libfftw3-dev" "libboost-program-options-dev" )

  elevate_if_not_root apt update

  # Dependencies required by main library
  for package in "${library_dependencies_ubuntu[@]}"; do
    if [[ $(dpkg-query --show --showformat='${db:Status-Abbrev}\n' ${package} 2> /dev/null | grep -q "ii"; echo $?) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root apt install -y --no-install-recommends ${package}
    fi
  done

  # Dependencies required by library client apps
  if [[ "${build_clients}" == true ]]; then
    for package in "${client_dependencies_ubuntu[@]}"; do
      if [[ $(dpkg-query --show --showformat='${db:Status-Abbrev}\n' ${package} 2> /dev/null | grep -q "ii"; echo $?) -ne 0 ]]; then
        printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
        elevate_if_not_root apt install -y --no-install-recommends ${package}
      fi
    done

    # The following builds googletest & lapack from source, installs into cmake default /usr/local
    pushd .
      printf "\033[32mBuilding \033[33mgoogletest & lapack\033[32m from source; installing into \033[33m/usr/local\033[0m\n"
      mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
      cmake -DBUILD_BOOST=OFF ../../deps
      make -j$(nproc)
      elevate_if_not_root make install
    popd
  fi

fi

export PATH=${PATH}:/opt/rocm/bin

pushd .
  # #################################################
  # configure & build
  # #################################################
  mkdir -p ${build_dir}/release/clients && cd ${build_dir}/release

  # On ROCm platforms, hcc compiler can build everything
  if [[ "${build_cuda}" == false ]]; then
    if [[ "${build_clients}" == true ]]; then
      CXX=hcc cmake -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SELFTEST=ON -DBUILD_CLIENTS_RIDER=ON ../..
    else
      CXX=hcc cmake ../..
    fi

    make -j$(nproc)
  else
    # The nvidia compile is a little more complicated, in that we split compiling the library from the clients
    # We use the hipcc compiler to build the rocfft library for a cuda backend (hipcc offloads the compile to nvcc)
    # However, we run into a compiler incompatibility compiling the clients between nvcc and fftw3.h 3.3.4 headers.
    # The incompatibility is fixed in fft v3.3.6, but that is not shipped by default on Ubuntu
    # As a workaround, since clients do not contain device code, we opt to build clients with the native
    # compiler on the platform.  The compiler cmake chooses during configuration time is mostly unchangeable,
    # so we launch multiple cmake invocation with a different compiler on each.

    # Build library only with hipcc as compiler
    CXX=hipcc cmake -DCMAKE_INSTALL_PREFIX=rocfft-install -DCPACK_PACKAGE_INSTALL_DIRECTORY=/opt/rocm ../..
    make -j$(nproc) install

    # Build cuda clients with default host compiler
    if [[ "${build_clients}" == true ]]; then
      pushd clients
        cmake -DCMAKE_PREFIX_PATH=$(pwd)/../rocfft-install -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SELFTEST=ON -DBUILD_CLIENTS_RIDER=ON ../../../clients
        make -j$(nproc)
      popd
    fi
  fi

  # #################################################
  # install
  # #################################################
  # installing through package manager, which makes uninstalling easy
  if [[ "${install_package}" == true ]]; then
    make package
    elevate_if_not_root dpkg -i rocfft-*.deb
  fi
popd