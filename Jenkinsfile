node('rocm') {
  //     sh 'env | sort'
    def scm_dir = pwd()
    def build_dir_debug = "${scm_dir}/../build/debug"
    def build_dir_release = "${scm_dir}/../build/release"
    dir("${scm_dir}") {
      stage 'Clone'
      checkout scm
    }
    dir("${build_dir_release}") {
      stage 'configure clang release'
        sh "cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_LIBRARY=ON -DBUILD_CLIENTS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DHIP_ROOT=/opt/rocm/hip -DBOOST_ROOT=/opt/boost/clang -DFFTW_ROOT=/usr/lib ${scm_dir}"
      stage 'Build'
        sh 'make -j 8'
      stage 'Package Debian'
        sh 'cd library-build; make package'
        archive includes: 'library-build/*.deb'
      stage 'samples'
        sh "cd clients-build/samples-build/fixed-16; ./fixed-16"
    }
    dir("${build_dir_debug}") {
      stage 'clang-tidy checks'
        sh "cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_LIBRARY=ON -DHIP_ROOT=/opt/rocm/hip -DCMAKE_CXX_CLANG_TIDY=\"clang-tidy-3.5;-checks=*\" ${scm_dir}"
        sh 'make'
    }
}
