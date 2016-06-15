node('rocm') {
  //     sh 'env | sort'
    def scm_dir = pwd()
    def build_dir_debug = "${scm_dir}/../build/debug"
    def build_dir_release = "${scm_dir}/../build/release"
    dir("${scm_dir}") {
      stage 'Clone'
      checkout scm
    }
    dir("${build_dir_debug}") {
      stage 'Configure Debug'
        sh "cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_LIBRARY=ON -DBUILD_CLIENTS=ON -DBUILD_CLIENTS_DEPENDENCY_BOOST=ON -DBUILD_CLIENTS_DEPENDENCY_GTEST=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON ${scm_dir}"
      stage 'Build'
        sh 'make -j 8'
      stage 'Package Debian'
        sh 'cd rocfft-build; make package'
        archive includes: 'rocfft-build/*.deb'
    }
}
