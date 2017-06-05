#!/usr/bin/env groovy

// Generated from snippet generator 'properties; set job properties'
// Keep only the most recent XX builds
properties([buildDiscarder(logRotator(
    artifactDaysToKeepStr: '',
    artifactNumToKeepStr: '',
    daysToKeepStr: '',
    numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
  ])

def email_list = "kent.knox@amd.com, bragadeesh.natarajan@amd.com, tingxing.dong@amd.com"
// def email_list = emailextrecipients([ [$class: 'CulpritsRecipientProvider'] ])

////////////////////////////////////////////////////////////////////////
// This encapsulates the cmake configure and build commands
def clang_build( String build_type, String clang_version, String boost_path, String src_path )
{
  // create softlinks for clang
  sh "sudo update-alternatives --install /usr/bin/clang  clang  /usr/bin/clang-${clang_version} 50 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-${clang_version}"

  stage("configure clang ${build_type}") {
    sh "cmake -DCMAKE_BUILD_TYPE=${build_type} -DBUILD_LIBRARY=ON -DBUILD_CLIENTS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_RIDER=ON -DHIP_ROOT=/opt/rocm/hip -DBOOST_ROOT=${boost_path} -DFFTW_ROOT=/usr/lib ${src_path}"
  }

  stage("Build") {
    sh "make -j\$(nproc)"
  }

  return void
}

////////////////////////////////////////////////////////////////////////
// This encapsulates running of unit tests
def run_tests(  )
{
  stage("unit tests") {
    sh '''
        cd clients-build/tests-build/staging
        ./rocfft-test-correctness --gtest_filter=*single* --gtest_output=xml
    '''
    junit 'clients-build/tests-build/staging/*.xml'
  }

  stage("rider") {
    sh '''
        cd clients-build/rider-build/staging
        ./rocfft-rider -x 16
        ./rocfft-rider -x 256
        ./rocfft-rider -x 1024
    '''
  }

  // stage("samples") {
  //   sh '''
  //       cd clients-build/samples-build/fixed-16
  //       ./fixed-16
  //     '''
  // }

  return void
}

def rocfft_build_pipeline( String build_type, String clang_version, String boost_path )
{
  String scm_dir = pwd()
  String build_dir_debug = "${scm_dir}/../build/debug"
  String build_dir_release = "${scm_dir}/../build/release"

  try
  {
    dir("${scm_dir}") {
      stage("Clone") {
        checkout scm
      }
    }

    dir("${build_dir_release}")
    {
      clang_build( "${build_type}", "${clang_version}", "${boost_path}", "${scm_dir}" )

      stage("Package Debian")
      {
        sh 'cd library-build; make package'
        archiveArtifacts artifacts: 'library-build/*.deb', fingerprint: true
        archiveArtifacts artifacts: 'library-build/*.rpm', fingerprint: true
        sh "sudo dpkg -c library-build/*.deb"
      }

      run_tests( )
    }
  }
  catch( err )
  {
    currentBuild.result = 'FAILURE'
    mail  to: "${email_list}",
          subject: "${env.JOB_NAME} finished with ${currentBuild.result}",
          body: "Node: ${env.NODE_NAME}\nSee ${env.BUILD_URL}\n\n" + err.toString()

    throw err
  }

  return void
}

node('rocm-1.5 && gfx803')
{
  rocfft_build_pipeline( "Release", "3.8", "/opt/boost/clang-3.8" )
}

////////////////////////////////////////////////////////////////////////
// node('rocm-1.4 && gfx803')
// {
//   rocfft_build_pipeline( "Release", "3.5", "/opt/boost" )
// }
