#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runTestCommand (platform, project, boolean debug=false)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    String testBinaryName = debug ? 'rocfft-test-d' : 'rocfft-test'
    String directory = debug ? 'debug' : 'release'

    def reference = (env.BRANCH_NAME ==~ /PR-\d+/) ? 'develop' : 'master'
    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, reference, false)
        }
    }

    def depCommand = """#!/usr/bin/env bash
                    set -x
                    ${getDependenciesCommand}
                    """
    platform.runCommand(this, depCommand)

    def dataTypes = ['single', 'double']
    for (def dataType in dataTypes)
    {
        def command = """#!/usr/bin/env bash
                    set -x
                    pwd
                    cd ${project.paths.project_build_prefix}
                    ./scripts/perf/alltime.py -i /opt/rocm/rocfft/clients/rocfft-rider -o ./${dataType}Ref -g 0 -p $dataType
                    ./scripts/perf/alltime.py -i ./build/${directory}/clients/staging/rocfft-rider -o ./${dataType}Change -g 0 -p $dataType
                    mkdir ${dataType}Results
                    ./scripts/perf/html_report.py ./${dataType}Ref ./${dataType}Change ./${dataType}Results
                    ls \$dataType
                    mv ${dataType}Results/figs.html ${dataType}Results/figs_${platform.gpu}.html
                """
         platform.runCommand(this, command)
         
         archiveArtifacts "${project.paths.project_build_prefix}/singleResults/*.html"
         publishHTML([allowMissing: false,
                    alwaysLinkToLastBuild: false,
                    keepAll: false,
                    reportDir: "${project.paths.project_build_prefix}/${dataType}Results",
                    reportFiles: "figs_${platform.gpu}.html",
                    reportName: "${dataType} Precision ${platform.gpu}",
                    reportTitles: "${dataType} Precision${platform.gpu}"])
    }
}

def runCI = 
{
    nodeDetails, jobName->

    def prj  = new rocProject('rocFFT-internal', 'PreCheckin')

    prj.defaults.ccache = true
    prj.timeout.compile = 600
    prj.timeout.test = 600
    prj.libraryDependencies = ['rocFFT-internal']

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false

    def commonGroovy

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project, jobName)
    }

    def testCommand =
    {
        platform, project->

        runTestCommand(platform, project)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, null)
}

ci: { 
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 1 * * 0')])]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["compute-rocm-dkms-no-npi-hipclang":([ubuntu18:['gfx900','gfx906']])]
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    propertyList.each 
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    jobNameList.each 
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(jobName) {
                runCI(nodeDetails, jobName)
            }
    }

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        stage(urlJobName) {
            runCI([ubuntu18:['gfx906']], urlJobName)
        }
    }
}
