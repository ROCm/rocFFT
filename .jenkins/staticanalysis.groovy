#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCompileCommand(platform, project, jobName, boolean debug=false)
{
    project.paths.construct_build_prefix()

    def command = """#!/usr/bin/env bash
            set -x
            ${project.paths.project_build_prefix}/docs/run_doc.sh
            """

    platform.runCommand(this, command)

    def yapfCommand = """#!/usr/bin/env bash
                         set -x
                         cd ${project.paths.project_build_prefix}
                         yapf --version
                         find . -iname '*.py' \
                         | grep -v 'build/'  \
                         | xargs -n 1 -P 1 -I{} -t sh -c 'yapf --style pep8 {} | diff - {}'
                      """

    platform.runCommand(this, yapfCommand)
    
    publishHTML([allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: false,
                reportDir: "${project.paths.project_build_prefix}/docs/source/_build/html",
                reportFiles: "index.html",
                reportName: "Documentation",
                reportTitles: "Documentation"])
}

def runCI =
{
    nodeDetails, jobName->

    def prj  = new rocProject('rocFFT-internal', 'StaticAnalysis')

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = true
    boolean staticAnalysis = true

    def compileCommand =
    {
        platform, project->

        runCompileCommand(platform, project, jobName, false)
    }

    buildProject(prj , formatCheck, nodes.dockerArray, compileCommand, null, null, staticAnalysis)


    def kernelSubsetPrj  = new rocProject('rocFFT-internal', 'BuildKernelSubset')

    def nodesForPrj2 = new dockerNodes(nodeDetails, jobName, kernelSubsetPrj)

    def commonGroovy

    def compileSubsetCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"

        // build pattern pow2,pow7 no manual small and large, dp only
        commonGroovy.runSubsetBuildCommand(platform, project, jobName, 'pow2,pow7', null, null, true)

        // build large sizes, dp only
        commonGroovy.runSubsetBuildCommand(platform, project, jobName, 'large', null, null, true)

        // build 2D sizes, dp only
        commonGroovy.runSubsetBuildCommand(platform, project, jobName, '2D', null, null, true)

        // put an extra unsupported size(10) in manual large to see if it will be filtered correctly
        commonGroovy.runSubsetBuildCommand(platform, project, jobName, 'none', null, '10,50,100,200,336', true)

        // put an extra unsupported size(23) in manual small to see if it will be filtered correctly
        commonGroovy.runSubsetBuildCommand(platform, project, jobName, 'none', '23,1024', '10,50,100,200,336', true)

        // all the manual sizes are not supported
        //commonGroovy.runSubsetBuildCommand(platform, project, jobName, 'none', '23', '10', true)
    }

    buildProject(kernelSubsetPrj , formatCheck, nodesForPrj2.dockerArray, compileSubsetCommand, null, null)
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * 6')])]))
    stage(urlJobName) {
        runCI([ubuntu20:['any']], urlJobName)
    }
}
