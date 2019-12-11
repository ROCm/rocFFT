#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
// Mostly generated from snippet generator 'properties; set job properties'
// Time-based triggers added to execute nightly tests, eg '30 2 * * *' means 2:30 AM
properties([
//    pipelineTriggers([cron('0 1 * * *'), [$class: 'PeriodicFolderTrigger', interval: '5m']]),
    buildDiscarder(logRotator(
      artifactDaysToKeepStr: '',
      artifactNumToKeepStr: '',
      daysToKeepStr: '',
      numToKeepStr: '10')),
    disableConcurrentBuilds(),
    // parameters([booleanParam( name: 'push_image_to_docker_hub', defaultValue: false, description: 'Push rocfft image to rocm docker-hub' )]),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])

////////////////////////////////////////////////////////////////////////
// import hudson.FilePath;
import java.nio.file.Path;

rocFFTCI:
{

    def rocfft = new rocProject('rocFFT-internal')

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['gfx900 && ubuntu', 'gfx906 && centos7', 'gfx900 && sles', 'gfx906 && sles', 'gfx900 && centos7','gfx906 && ubuntu && hip-clang'], rocfft)

    boolean formatCheck = true

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()

        String compiler = platform.jenkinsLabel.contains('hip-clang') ? 'hipcc' : 'hcc'
        String clientArgs = '-DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SELFTEST=ON -DBUILD_CLIENTS_RIDER=ON'
        String hipClangArgs = platform.jenkinsLabel.contains('hip-clang') ? '-DUSE_HIP_CLANG=ON -DHIP_COMPILER=clang' : ''
        String cmake = platform.jenkinsLabel.contains('centos') ? 'cmake3' : 'cmake'
        String sudo = platform.jenkinsLabel.contains('sles') ? 'sudo' : ''

        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    mkdir build && cd build
                    ${sudo} ${cmake} -DCMAKE_CXX_COMPILER=/opt/rocm/bin/${compiler} ${clientArgs} ${hipClangArgs} ..
                    ${sudo} make -j\$(nproc)
                """
        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->
            
        String sudo = auxiliary.sudo(platform.jenkinsLabel)

        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/clients/staging
                    ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocfft-test --gtest_color=yes
                """
        platform.runCommand(this, command)
    }

    def packageCommand =
    {
        platform, project->

        if(platform.jenkinsLabel.contains('hip-clang'))
        {
            packageCommand = null
        }
        else
        {
            def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build",true,true)
            platform.runCommand(this, packageHelper[0])
            platform.archiveArtifacts(this, packageHelper[1])
        }
    }

    buildProject(rocfft, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}
