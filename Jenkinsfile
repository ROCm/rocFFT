#!/usr/bin/env groovy

// Generated from snippet generator 'properties; set job properties'
properties([buildDiscarder(logRotator(
    artifactDaysToKeepStr: '',
    artifactNumToKeepStr: '',
    daysToKeepStr: '',
    numToKeepStr: '10')),
    disableConcurrentBuilds(),
    // parameters([booleanParam( name: 'push_image_to_docker_hub', defaultValue: false, description: 'Push rocfft image to rocm docker-hub' )]),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])

////////////////////////////////////////////////////////////////////////
// -- AUXILLARY HELPER FUNCTIONS
// import hudson.FilePath;
import java.nio.file.Path;

////////////////////////////////////////////////////////////////////////
// Return build number of upstream job
@NonCPS
int get_upstream_build_num( )
{
    def upstream_cause = currentBuild.rawBuild.getCause( hudson.model.Cause$UpstreamCause )
    if( upstream_cause == null)
      return 0

    return upstream_cause.getUpstreamBuild()
}

////////////////////////////////////////////////////////////////////////
// Return project name of upstream job
@NonCPS
String get_upstream_build_project( )
{
    def upstream_cause = currentBuild.rawBuild.getCause( hudson.model.Cause$UpstreamCause )
    if( upstream_cause == null)
      return null

    return upstream_cause.getUpstreamProject()
}

////////////////////////////////////////////////////////////////////////
// Calculate the relative path between two sub-directories from a common root
@NonCPS
String g_relativize( String root_string, String rel_source, String rel_build )
{
  Path root_path = new File( root_string ).toPath( )
  Path path_src = root_path.resolve( rel_source )
  Path path_build = root_path.resolve( rel_build )

  return path_build.relativize( path_src ).toString( )
}

////////////////////////////////////////////////////////////////////////
// Construct the relative path of the build directory
void build_directory_rel( project_paths paths, compiler_data hcc_args )
{
  // if( hcc_args.build_config.equalsIgnoreCase( 'release' ) )
  // {
  //   paths.project_build_prefix = paths.build_prefix + '/' + paths.project_name + '/release';
  // }
  // else
  // {
  //   paths.project_build_prefix = paths.build_prefix + '/' + paths.project_name + '/debug';
  // }
  paths.project_build_prefix = paths.build_prefix + '/' + paths.project_name;

}

////////////////////////////////////////////////////////////////////////
// Lots of images are created above; no apparent way to delete images:tags with docker global variable
def docker_clean_images( String org, String image_name )
{
  // Check if any images exist first grepping for image names
  int docker_images = sh( script: "docker images | grep \"${org}/${image_name}\"", returnStatus: true )

  // The script returns a 0 for success (images were found )
  if( docker_images == 0 )
  {
    // run bash script to clean images:tags after successful pushing
    sh "docker images | grep \"${org}/${image_name}\" | awk '{print \$1 \":\" \$2}' | xargs docker rmi"
  }
}

////////////////////////////////////////////////////////////////////////
// -- BUILD RELATED FUNCTIONS

////////////////////////////////////////////////////////////////////////
// Checkout source code, source dependencies and update version number numbers
// Returns a relative path to the directory where the source exists in the workspace
void checkout_and_version( project_paths paths )
{
  paths.project_src_prefix = paths.src_prefix + '/' + paths.project_name

  dir( paths.project_src_prefix )
  {
    // checkout rocfft
    checkout([
      $class: 'GitSCM',
      branches: scm.branches,
      doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
      extensions: scm.extensions + [[$class: 'CleanCheckout']],
      userRemoteConfigs: scm.userRemoteConfigs
    ])

    if( fileExists( 'CMakeLists.txt' ) )
    {
      def cmake_version_file = readFile( 'CMakeLists.txt' ).trim()
      //echo "cmake_version_file:\n${cmake_version_file}"

      cmake_version_file = cmake_version_file.replaceAll(/(\d+\.)(\d+\.)(\d+\.)\d+/, "\$1\$2\$3${env.BUILD_ID}")
      //echo "cmake_version_file:\n${cmake_version_file}"
      writeFile( file: 'CMakeLists.txt', text: cmake_version_file )
    }
  }

}

////////////////////////////////////////////////////////////////////////
// This creates the docker image that we use to build the project in
// The docker images contains all dependencies, including OS platform, to build
def docker_build_image( docker_data docker_args, project_paths paths )
{
  String build_image_name = "build"
  def build_image = null

  dir( paths.project_src_prefix )
  {
    def user_uid = sh( script: 'id -u', returnStdout: true ).trim()

    // Docker 17.05 introduced the ability to use ARG values in FROM statements
    // Docker inspect failing on FROM statements with ARG https://issues.jenkins-ci.org/browse/JENKINS-44836
    // build_image = docker.build( "${paths.project_name}/${build_image_name}:latest", "--pull -f docker/${build_docker_file} --build-arg user_uid=${user_uid} --build-arg base_image=${from_image} ." )

    // JENKINS-44836 workaround by using a bash script instead of docker.build()
    sh "docker build -t ${paths.project_name}/${build_image_name}:latest -f docker/${docker_args.build_docker_file} ${docker_args.docker_build_args} --build-arg user_uid=${user_uid} --build-arg base_image=${docker_args.from_image} ."
    build_image = docker.image( "${paths.project_name}/${build_image_name}:latest" )
  }

  return build_image
}

////////////////////////////////////////////////////////////////////////
// This encapsulates the cmake configure, build and package commands
// Leverages docker containers to encapsulate the build in a fixed environment
Boolean docker_build_inside_image( def build_image, compiler_data compiler_args, docker_data docker_args, project_paths paths )
{
  // Construct a relative path from build directory to src directory; used to invoke cmake
  String rel_path_to_src = g_relativize( pwd( ), paths.project_src_prefix, paths.project_build_prefix )

  String build_type_postfix = null
  if( compiler_args.build_config.equalsIgnoreCase( 'release' ) )
  {
    build_type_postfix = ""
  }
  else
  {
    build_type_postfix = "-d"
  }

  // For the nvidia path, we somewhat arbitrarily choose to use the hcc-ctu rocblas package
  String rocblas_archive_path=compiler_args.compiler_name;
  if( rocblas_archive_path.toLowerCase( ).startsWith( 'nvcc-' ) )
  {
    rocblas_archive_path='hcc-ctu'
  }

  build_image.inside( docker_args.docker_run_args )
  {
    withEnv(["CXX=${compiler_args.compiler_path}", 'CLICOLOR_FORCE=1'])
    {
      // Build library & clients
      sh  """#!/usr/bin/env bash
          set -x
          cd ${paths.project_build_prefix}
          ${paths.build_command}
        """
    }

    stage( "Test ${compiler_args.compiler_name} ${compiler_args.build_config}" )
    {
      // Cap the maximum amount of testing to be a few hours; assume failure if the time limit is hit
      timeout(time: 1, unit: 'HOURS')
      {
        sh """#!/usr/bin/env bash
              set -x
              cd ${paths.project_build_prefix}/build/release/clients/staging
              ./rocfft-test${build_type_postfix} --gtest_output=xml --gtest_color=yes
          """
        junit "${paths.project_build_prefix}/build/release/clients/staging/*.xml"
      }

      String docker_context = "${compiler_args.build_config}/${compiler_args.compiler_name}"
      if( compiler_args.compiler_name.toLowerCase( ).startsWith( 'hcc-' ) )
      {
        sh  """#!/usr/bin/env bash
            set -x
            cd ${paths.project_build_prefix}/build/release
            make package
          """

        sh  """#!/usr/bin/env bash
            set -x
            rm -rf ${docker_context} && mkdir -p ${docker_context}
            mv ${paths.project_build_prefix}/build/release/*.deb ${docker_context}
            # mv ${paths.project_build_prefix}/build/release/*.rpm ${docker_context}
            dpkg -c ${docker_context}/*.deb
        """

        archiveArtifacts artifacts: "${docker_context}/*.deb", fingerprint: true
        // archiveArtifacts artifacts: "${docker_context}/*.rpm", fingerprint: true
      }
    }
  }

  return true
}

////////////////////////////////////////////////////////////////////////
// This builds a fresh docker image FROM a clean base image, with no build dependencies included
// Uploads the new docker image to internal artifactory
// String docker_test_install( String hcc_ver, String artifactory_org, String from_image, String rocfft_src_rel, String build_dir_rel )
String docker_test_install( compiler_data compiler_args, docker_data docker_args, project_paths rocfft_paths, String job_name )
{
  def rocfft_install_image = null
  String image_name = "rocfft-hip-${compiler_args.compiler_name}-ubuntu-16.04"
  String docker_context = "${compiler_args.build_config}/${compiler_args.compiler_name}"

  stage( "Artifactory ${compiler_args.compiler_name} ${compiler_args.build_config}" )
  {
    //  We copy the docker files into the bin directory where the .deb lives so that it's a clean build everytime
    sh  """#!/usr/bin/env bash
        set -x
        mkdir -p ${docker_context}
        cp -r ${rocfft_paths.project_src_prefix}/docker/* ${docker_context}
      """

    // Docker 17.05 introduced the ability to use ARG values in FROM statements
    // Docker inspect failing on FROM statements with ARG https://issues.jenkins-ci.org/browse/JENKINS-44836
    // rocfft_install_image = docker.build( "${job_name}/${image_name}:${env.BUILD_NUMBER}", "--pull -f ${build_dir_rel}/dockerfile-rocfft-ubuntu-16.04 --build-arg base_image=${from_image} ${build_dir_rel}" )

    // JENKINS-44836 workaround by using a bash script instead of docker.build()
    sh """docker build -t ${job_name}/${image_name} --pull -f ${docker_context}/${docker_args.install_docker_file} \
        --build-arg base_image=${docker_args.from_image} ${docker_context}"""
    rocfft_install_image = docker.image( "${job_name}/${image_name}" )
  }

  return image_name
}

// Docker related variables gathered together to reduce parameter bloat on function calls
class docker_data implements Serializable
{
  String from_image
  String build_docker_file
  String install_docker_file
  String docker_run_args
  String docker_build_args
}

// Docker related variables gathered together to reduce parameter bloat on function calls
class compiler_data implements Serializable
{
  String compiler_name
  String build_config
  String compiler_path
}

// Paths variables bundled together to reduce parameter bloat on function calls
class project_paths implements Serializable
{
  String project_name
  String src_prefix
  String project_src_prefix
  String build_prefix
  String project_build_prefix
  String build_command
}

////////////////////////////////////////////////////////////////////////
// -- MAIN
// Following this line is the start of MAIN of this Jenkinsfile

// sh  """
//     set -x
// # printf '\033[31mHello World\033[0m'
// # echo "TERM=${env.TERM}"
// # echo "LANG=${env.LANG}"
// # echo "SHELL=${env.SHELL}"
// """

// Integration testing is a special path which implies testing of an upsteam build of hcc,
// but does not need testing across older builds of hcc or cuda.
// params.hip_integration_test is set in HIP build
// NOTE: hip does not currently set this bit; this is non-functioning at this time
// if( params.hip_integration_test )
// {
//   println "Enabling rocfft integration testing pass"

//   node('docker && rocm')
//   {
//     rocfft_integration_testing( '--device=/dev/kfd', 'hip-ctu', 'Release' )
//   }

//   return
// }

// This defines a common build pipeline used by most targets
def build_pipeline( compiler_data compiler_args, docker_data docker_args, project_paths rocfft_paths, def docker_inside_closure )
{
  ansiColor( 'vga' )
  {
    // NOTE: build_succeeded does not appear to be local to each function invokation.  I couldn't use it where each
    // node had a different success value.
    def build_succeeded = false;

    stage( "Build ${compiler_args.compiler_name} ${compiler_args.build_config}" )
    {
      // Checkout source code, dependencies and version files
      checkout_and_version( rocfft_paths )

      // Conctruct a binary directory path based on build config
      build_directory_rel( rocfft_paths, compiler_args );

      // Create/reuse a docker image that represents the rocfft build environment
      def rocfft_build_image = docker_build_image( docker_args, rocfft_paths )

      // Print system information for the log
      rocfft_build_image.inside( docker_args.docker_run_args, docker_inside_closure )

      // Build rocfft inside of the build environment
      build_succeeded = docker_build_inside_image( rocfft_build_image, compiler_args, docker_args, rocfft_paths )
    }

    // After a successful build, test the installer
    // Only do this for rocm based builds
    if( compiler_args.compiler_name.toLowerCase( ).startsWith( 'hcc-' ) )
    {
      String job_name = env.JOB_NAME.toLowerCase( )
      String rocfft_image_name = docker_test_install( compiler_args, docker_args, rocfft_paths, job_name )

      docker_clean_images( job_name, rocfft_image_name )
    }
  }
}

// The following launches 3 builds in parallel: hcc-ctu, hcc-1.6 and cuda
parallel hcc_ctu:
{
  node( 'docker && rocm && dkms' )
  {
    def docker_args = new docker_data(
        from_image:'compute-artifactory:5001/rocm-developer-tools/hip/master/hip-hcc-ctu-ubuntu-16.04:latest',
        build_docker_file:'dockerfile-build-ubuntu-16.04',
        install_docker_file:'dockerfile-rocfft-ubuntu-16.04',
        docker_run_args:'--device=/dev/kfd --device=/dev/dri --group-add=video',
        docker_build_args:' --pull' )

    def compiler_args = new compiler_data(
        compiler_name:'hcc-ctu',
        build_config:'Release',
        compiler_path:'/opt/rocm/bin/hcc' )

    def rocfft_paths = new project_paths(
        project_name:'rocfft-hcc-ctu',
        src_prefix:'src',
        build_prefix:'src',
        build_command: './install.sh -cd' )

    def print_version_closure = {
      sh  """
          set -x
          /opt/rocm/bin/rocm_agent_enumerator -t ALL
          /opt/rocm/bin/hcc --version
        """
    }

    build_pipeline( compiler_args, docker_args, rocfft_paths, print_version_closure )
  }
},
hcc_rocm:
{
  node( 'docker && rocm && dkms' )
  {
    def hcc_docker_args = new docker_data(
        from_image:'rocm/dev-ubuntu-16.04:1.8.0',
        build_docker_file:'dockerfile-build-ubuntu-16.04',
        install_docker_file:'dockerfile-rocfft-ubuntu-16.04',
        docker_run_args:'--device=/dev/kfd --device=/dev/dri --group-add=video',
        docker_build_args:' --pull' )

    def hcc_compiler_args = new compiler_data(
        compiler_name:'hcc-rocm',
        build_config:'Release',
        compiler_path:'/opt/rocm/bin/hcc' )

    def rocfft_paths = new project_paths(
        project_name:'rocfft-hcc-rocm',
        src_prefix:'src',
        build_prefix:'src',
        build_command: './install.sh -cd' )

    def print_version_closure = {
      sh  """
          set -x
          /opt/rocm/bin/rocm_agent_enumerator -t ALL
          /opt/rocm/bin/hcc --version
        """
    }

    build_pipeline( hcc_compiler_args, hcc_docker_args, rocfft_paths, print_version_closure )
  }
} //,
// nvcc:
// {
//   node( 'docker && cuda' )
//   {
//     def hcc_docker_args = new docker_data(
//         from_image:'nvidia/cuda:9.1-devel-ubuntu16.04',
//         build_docker_file:'dockerfile-build-nvidia-cuda',
//         install_docker_file:'dockerfile-rocfft-ubuntu-cuda',
//         docker_run_args:'--runtime=nvidia',
//         docker_build_args:' --pull' )

//     def hcc_compiler_args = new compiler_data(
//         compiler_name:'nvcc-9.1',
//         build_config:'Release',
//         compiler_path:'g++' )

//     def rocfft_paths = new project_paths(
//         project_name:'rocfft-nvcc',
//         src_prefix:'src',
//         build_prefix:'src',
//         build_command: './install.sh -cd --cuda' )

//     def print_version_closure = {
//       sh  """
//           set -x
//           nvidia-smi
//           nvcc --version
//         """
//     }

//     build_pipeline( hcc_compiler_args, hcc_docker_args, rocfft_paths, print_version_closure )
//   }
// }
