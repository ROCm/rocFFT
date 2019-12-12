// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project)
{
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

def runTestCommand (platform, project)
{            
    String sudo = auxiliary.sudo(platform.jenkinsLabel)

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/clients/staging
                ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocfft-test --gtest_color=yes
            """
    platform.runCommand(this, command)
}

def runPackageCommand(platform, project)
{
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

return this
