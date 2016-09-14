#!/usr/bin/env groovy

currentBuild.result = "SUCCESS"
node('rocm && fiji')
{
  //     sh 'env | sort'
    def scm_dir = pwd()
    def build_dir_debug = "${scm_dir}/../build/debug"
    def build_dir_release = "${scm_dir}/../build/release"

    try
    {
      dir("${scm_dir}") {
        stage("Clone") {
          checkout scm
        }
      }

      sh "sudo update-alternatives --install /usr/bin/clang  clang  /usr/bin/clang-3.5 50 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-3.5"

      dir("${build_dir_release}") {
        // create softlinks for clang
        stage("configure clang release") {
          sh "cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_LIBRARY=ON -DBUILD_CLIENTS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DHIP_ROOT=/opt/rocm/hip -DBOOST_ROOT=/opt/boost/clang -DFFTW_ROOT=/usr/lib ${scm_dir}"
        }

        stage("Build") {
          sh 'make -j 8'
        }

        stage("Package Debian") {
          sh 'cd library-build; make package'
          archive includes: 'library-build/*.deb'
        }

        stage("samples") {
          sh "cd clients-build/samples-build/fixed-16; ./fixed-16"
        }

        stage("unit tests") {
          // junit 'test_detail.xml'
        }
      }

      dir("${build_dir_debug}") {
        stage("clang-tidy checks") {
          sh "cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_LIBRARY=ON -DHIP_ROOT=/opt/rocm/hip -DCMAKE_CXX_CLANG_TIDY=\"clang-tidy-3.5;-checks=*\" ${scm_dir}"
          sh 'make'
        }
      }
    }
    catch( err )
    {
        currentBuild.result = "FAILURE"

        // emailext pipeline projects do not seem to support token expansions yet, such as ${env.JOB_NAME}
        // emailext recipientProviders: [[$class: 'CulpritsRecipientProvider']], to: 'dl.library-team@amd.com'

        // mail job does not appear to work with emailextrecipients
        def email_list = emailextrecipients([
                [$class: 'CulpritsRecipientProvider']
        ])

        mail  to: 'kent.knox@amd.com, bragadeesh.natarajan@amd.com, tingxing.dong@amd.com',
              subject: "${env.JOB_NAME} finished with ${currentBuild.result}",
              body: "Node: ${env.NODE_NAME}\nSee ${env.BUILD_URL}\n"
        throw err
    }
}
