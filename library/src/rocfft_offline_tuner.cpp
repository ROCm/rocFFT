// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../../shared/environment.h"
#include "../../shared/gpubuf.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/rocfft_params.h"
#include "option_util.h"
#include "rocfft.h"
#include "tuning_helper.h"

inline void
    hip_V_Throw(hipError_t res, const std::string& msg, size_t lineno, const std::string& fileName)
{
    if(res != hipSuccess)
    {
        std::stringstream tmp;
        tmp << "HIP_V_THROWERROR< ";
        tmp << res;
        tmp << " > (";
        tmp << fileName;
        tmp << " Line: ";
        tmp << lineno;
        tmp << "): ";
        tmp << msg;
        std::string errorm(tmp.str());
        std::cout << errorm << std::endl;
        throw std::runtime_error(errorm);
    }
}

inline void
    lib_V_Throw(fft_status res, const std::string& msg, size_t lineno, const std::string& fileName)
{
    if(res != fft_status_success)
    {
        std::stringstream tmp;
        tmp << "LIB_V_THROWERROR< ";
        tmp << res;
        tmp << " > (";
        tmp << fileName;
        tmp << " Line: ";
        tmp << lineno;
        tmp << "): ";
        tmp << msg;
        std::string errorm(tmp.str());
        std::cout << errorm << std::endl;
        throw std::runtime_error(errorm);
    }
}

#define HIP_V_THROW(_status, _message) hip_V_Throw(_status, _message, __LINE__, __FILE__)
#define LIB_V_THROW(_status, _message) lib_V_Throw(_status, _message, __LINE__, __FILE__)

static const int command_tuning  = 0;
static const int command_merging = 1;

int merge_solutions(const std::string& base_filename,
                    const std::string& new_filename,
                    const std::string& probKey,
                    const std::string& out_filename)
{
    // don't use anything from solutions.cpp
    rocfft_setenv("ROCFFT_USE_EMPTY_SOL_MAP", "1");

    rocfft_setup();

    // create tuning parameters
    TuningBenchmarker* offline_tuner = nullptr;
    rocfft_get_offline_tuner_handle((void**)(&offline_tuner));

    // Manupulating the solution map from tuner...
    bool merge_result
        = offline_tuner->MergingSolutionsMaps(base_filename, new_filename, probKey, out_filename);

    rocfft_cleanup();

    if(!merge_result)
    {
        std::cout << "Merge Solutions Failed" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int offline_tune_problems(rocfft_params& params, int verbose, int ntrial)
{
    // don't use anything from solutions.cpp
    rocfft_setenv("ROCFFT_USE_EMPTY_SOL_MAP", "1");

    rocfft_setup();

    params.validate();

    if(!params.valid(verbose))
        throw std::runtime_error("Invalid parameters, add --verbose=1 for detail");
    if(verbose)
        std::cout << params.str(" ") << std::endl;

    std::cout << "Token: " << params.token() << std::endl;

    // create tuning parameters
    TuningBenchmarker* offline_tuner = nullptr;
    rocfft_get_offline_tuner_handle((void**)(&offline_tuner));

    // first time call create_plan is actually generating a bunch of combination of configs
    offline_tuner->SetInitStep(0);

    // Check free and total available memory:
    size_t free  = 0;
    size_t total = 0;
    HIP_V_THROW(hipMemGetInfo(&free, &total), "hipMemGetInfo failed");
    const auto raw_vram_footprint
        = params.fft_params_vram_footprint() + twiddle_table_vram_footprint(params);
    if(!vram_fits_problem(raw_vram_footprint, free))
    {
        std::cout << "SKIPPED: Problem size (" << raw_vram_footprint
                  << ") raw data too large for device.\n";
        return EXIT_SUCCESS;
    }

    const auto vram_footprint = params.vram_footprint();
    if(!vram_fits_problem(vram_footprint, free))
    {
        std::cout << "SKIPPED: Problem size (" << vram_footprint
                  << ") raw data too large for device.\n";
        return EXIT_SUCCESS;
    }

    LIB_V_THROW(params.create_plan(), "Plan creation failed");

    // GPU input buffer:
    auto                ibuffer_sizes = params.ibuffer_sizes();
    std::vector<gpubuf> ibuffer(ibuffer_sizes.size());
    std::vector<void*>  pibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        HIP_V_THROW(ibuffer[i].alloc(ibuffer_sizes[i]), "Creating input Buffer failed");
        pibuffer[i] = ibuffer[i].data();
    }

    // Input data:
    params.compute_input(ibuffer);

    // GPU output buffer:
    std::vector<gpubuf>  obuffer_data;
    std::vector<gpubuf>* obuffer = &obuffer_data;
    if(params.placement == fft_placement_inplace)
    {
        obuffer = &ibuffer;
    }
    else
    {
        auto obuffer_sizes = params.obuffer_sizes();
        obuffer_data.resize(obuffer_sizes.size());
        for(unsigned int i = 0; i < obuffer_data.size(); ++i)
        {
            HIP_V_THROW(obuffer_data[i].alloc(obuffer_sizes[i]), "Creating output Buffer failed");
        }
    }
    std::vector<void*> pobuffer(obuffer->size());
    for(unsigned int i = 0; i < obuffer->size(); ++i)
    {
        pobuffer[i] = obuffer->at(i).data();
    }

    // finish initialization, solution map now contains all the candidates
    // start doing real benchmark with different configurations
    int num_nodes = offline_tuner->UpdateNumOfTuningNodes();
    if(num_nodes == 0)
    {
        std::cout << "[Result]: This fft problem hasn't been supported yet. (Prime number or "
                     "2D-Single)"
                  << std::endl;
        rocfft_cleanup();
        return EXIT_FAILURE;
    }

    static const double max_double = std::numeric_limits<double>().max();

    bool                     csv_is_created    = false;
    double                   overall_best_time = max_double;
    std::vector<int>         winner_phases     = std::vector<int>(num_nodes, 0);
    std::vector<int>         winner_ids        = std::vector<int>(num_nodes, 0);
    std::vector<std::string> kernels           = std::vector<std::string>(num_nodes, "");
    std::vector<double>      node_best_times   = std::vector<double>(num_nodes, max_double);

    // calculate this once only
    const double totsize
        = std::accumulate(params.length.begin(), params.length.end(), 1, std::multiplies<size_t>());
    const double k
        = ((params.itype == fft_array_type_real) || (params.otype == fft_array_type_real)) ? 2.5
                                                                                           : 5.0;
    const double opscount = (double)params.nbatch * k * totsize * log(totsize) / log(2.0);

    static const int TUNING_PHASE = 2;
    for(int curr_phase = 0; curr_phase < TUNING_PHASE; ++curr_phase)
    {
        if(curr_phase > 0)
        {
            // SET TARGET_FACTOR and current PHASE
            offline_tuner->SetInitStep(curr_phase);

            // make sure we can re-create the plan
            params.free();

            LIB_V_THROW(params.create_plan(), "Plan creation failed");
        }

        // keeping creating plan
        for(int node_id = 0; node_id < num_nodes; ++node_id)
        {
            std::string winner_name;
            int         winner_phase;
            int         winner_id;
            int         num_benchmarks = offline_tuner->GetNumOfKernelCandidates(node_id);

            offline_tuner->SetCurrentTuningNodeId(node_id);
            for(int ssn = 0; ssn < num_benchmarks; ++ssn)
            {
                offline_tuner->SetCurrentKernelCandidateId(ssn);
                std::cout << "\nTuning for node " << node_id << "/" << (num_nodes - 1)
                          << ", tuning phase :" << curr_phase << "/" << (TUNING_PHASE - 1)
                          << ", config :" << ssn << "/" << (num_benchmarks - 1) << std::endl;

                // make sure we can re-create the plan
                params.free();

                LIB_V_THROW(params.create_plan(), "Plan creation failed");

                // skip low occupancy test...simple output gflops 0, and a max double as ms
                BenchmarkInfo info = offline_tuner->GetCurrBenchmarkInfo();
                // we allow 2D_SINGLE kernels with occupancy 1
                if(info.threads_per_trans[1] != 0)
                {
                    if(info.occupancy < 0)
                    {
                        std::cout << "\nOccupancy -1 (unable to gen kernel), Skipped" << std::endl;
                        offline_tuner->UpdateCurrBenchResult(max_double, 0);
                        continue;
                    }
                }
                else
                {
                    if(info.occupancy == 1 || info.occupancy < 0)
                    {
                        std::cout << "\nOccupancy 1 or -1, Skipped" << std::endl;
                        offline_tuner->UpdateCurrBenchResult(max_double, 0);
                        continue;
                    }
                }

                params.execute(pibuffer.data(), pobuffer.data());

                // Run the transform several times and record the execution time:
                std::vector<double> gpu_time(ntrial);

                hipEvent_wrapper_t start, stop;
                start.alloc();
                stop.alloc();
                for(unsigned int itrial = 0; itrial < gpu_time.size(); ++itrial)
                {
                    HIP_V_THROW(hipEventRecord(start), "hipEventRecord failed");

                    params.execute(pibuffer.data(), pobuffer.data());

                    HIP_V_THROW(hipEventRecord(stop), "hipEventRecord failed");
                    HIP_V_THROW(hipEventSynchronize(stop), "hipEventSynchronize failed");

                    float time;
                    HIP_V_THROW(hipEventElapsedTime(&time, start, stop),
                                "hipEventElapsedTime failed");
                    gpu_time[itrial] = time;
                }

                std::cout << "Execution gpu time:";
                for(const auto& i : gpu_time)
                {
                    std::cout << " " << i;
                }
                std::cout << " ms" << std::endl;

                std::cout << "Execution gflops:  ";
                for(const auto& i : gpu_time)
                {
                    double gflops = opscount / (1e6 * i);
                    std::cout << " " << gflops;
                }
                std::cout << std::endl;

                // get median, if odd, get middle one, else get avg(middle twos)
                std::sort(gpu_time.begin(), gpu_time.end());
                double ms_median
                    = (gpu_time.size() % 2 == 1)
                          ? gpu_time[gpu_time.size() / 2]
                          : (gpu_time[gpu_time.size() / 2] + gpu_time[gpu_time.size() / 2 - 1]) / 2;
                double gflops_median = opscount / (1e6 * ms_median);

                offline_tuner->UpdateCurrBenchResult(ms_median, gflops_median);
                overall_best_time = std::min(overall_best_time, ms_median);
            }

            offline_tuner->FindWinnerForCurrNode(
                node_best_times[node_id], winner_phase, winner_id, winner_name);
            std::cout << "\n[UP_TO_PHASE_" << curr_phase << "_RESULT]:" << std::endl;
            std::cout << "\n[BEST_KERNEL]: In Phase: " << winner_phase
                      << ", Config ID: " << winner_id << std::endl;

            // update the latest winner info
            winner_phases[node_id] = winner_phase;
            winner_ids[node_id]    = winner_id;
            kernels[node_id]       = winner_name;

            bool is_last_phase = (curr_phase == TUNING_PHASE - 1);
            bool is_last_node  = (node_id == num_nodes - 1);

            // output data of this turn to csv
            csv_is_created = offline_tuner->ExportCSV(csv_is_created) || csv_is_created;
            if(!csv_is_created)
                std::cout << "CSV is not created or is written failed." << std::endl;

            // pass the target factors to next phase with permutation
            if(!is_last_phase)
                offline_tuner->PropagateBestFactorsToNextPhase();

            // in last phase and last node: finished tuning
            // export to file (output the winner solutions to solution map)
            if(is_last_phase && is_last_node)
                offline_tuner->ExportWinnerToSolutions();
        }
    }

    std::string out_path;
    offline_tuner->GetOutputSolutionMapPath(out_path);

    std::cout << "\n[OUTPUT_FILE]: " << out_path << std::endl;
    std::cout << "\n[BEST_SOLUTION]: " << params.token() << std::endl;
    for(int node_id = 0; node_id < num_nodes; ++node_id)
    {
        std::cout << "[Result]: Node " << node_id << ":" << std::endl;
        std::cout << "[Result]:     in phase   : " << winner_phases[node_id] << std::endl;
        std::cout << "[Result]:     best config: " << winner_ids[node_id] << std::endl;
        std::cout << "[Result]:     kernel name: " << kernels[node_id] << std::endl;
    }
    double best_gflops = opscount / (1e6 * overall_best_time);
    std::cout << "[Result]: GPU Time: " << overall_best_time << std::endl;
    std::cout << "[Result]: GFLOPS: " << best_gflops << std::endl;

    rocfft_cleanup();

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    // This helps with mixing output of both wide and narrow characters to the screen
    std::ios::sync_with_stdio(false);

    rocfft_params params;
    std::string   lengthArgStr;
    std::string   precisionStr;
    int           verbose;
    int           deviceId;
    int           ntrial;
    int           command_type; // 0: tuning , 1: merging

    int transform_type_int;
    int itype_int;
    int otype_int;

    std::string base_sol_filename   = "";
    std::string adding_sol_filename = "";
    std::string adding_problemkey   = "";
    std::string output_sol_filename = "";

    // Declare the supported options.
    // clang-format off
    options_description opdesc("rocfft offline tuner command line options");
    opdesc.add_options()("help,h", "produces this help message")
        ("version,v", "Print queryable version information from the rocfft library")
        ("command", value<int>(&command_type)->default_value(0), "Action to do:\n0) tuning\n1) merging solution map\n(default: 0)")

        ("base_sol_file", value<std::string>(&base_sol_filename), "filename of base-solution-map")
        ("new_sol_file", value<std::string>(&adding_sol_filename), "filename of new-solution-map")
        ("new_probkey", value<std::string>(&adding_problemkey), "problemkey (\"arch:token\") of the solution to be added, (looking up the new-solution-map)")
        ("output_sol_file", value<std::string>(&output_sol_filename), "filename of merged-solution-map")

        ("device", value<int>(&deviceId)->default_value(0), "Select a specific device id")
        ("verbose", value<int>(&verbose)->default_value(0), "Control output verbosity")
        ("ntrial,N", value<int>(&ntrial)->default_value(1), "Trial size for the problem")
        ("notInPlace,o", "Not in-place FFT transform (default: in-place)")
        ("precision", value<std::string>(&precisionStr), "Transform precision: single (default), double, half")
        ("transformType,t", value<int>(&transform_type_int)
         ->default_value((int)fft_transform_type_complex_forward),
         "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
         "forward\n3) real inverse")
        ( "batchSize,b", value<size_t>(&params.nbatch)->default_value(1),
          "If this value is greater than one, arrays will be used ")
        ( "itype", value<int>(&itype_int)
          ->default_value((int)fft_array_type_unset),
          "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ( "otype", value<int>(&otype_int)
          ->default_value((int)fft_array_type_unset),
          "Array type of output data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ("length",  value<std::string>(&lengthArgStr), "Lengths.(Separate by comma)");
    // clang-format on

    variables_map vm;
    store(parse_command_line(argc, argv, opdesc), vm);
    notify(vm);

    //
    // MERGING COMMAND
    //
    if(command_type == command_merging)
    {
        if(!vm.count("new_sol_file"))
        {
            std::cout << "Please specify file-path of the new solution map" << std::endl;
            return EXIT_FAILURE;
        }
        if(!vm.count("new_probkey"))
        {
            std::cout << "Please specify the problem-key to be added" << std::endl;
            return EXIT_FAILURE;
        }
        if(!vm.count("output_sol_file"))
        {
            std::cout << "Please specify file-path of the output solution map" << std::endl;
            return EXIT_FAILURE;
        }

        return merge_solutions(
            base_sol_filename, adding_sol_filename, adding_problemkey, output_sol_filename);
    }

    if(command_type != command_tuning)
    {
        std::cout << "Unknown command:" << command_type << std::endl;
        return EXIT_FAILURE;
    }

    //
    // TUNING COMMAND
    //
    if(vm.count("precision"))
    {
        if(precisionStr == "half")
            params.precision = fft_precision_half;
        else if(precisionStr == "single")
            params.precision = fft_precision_single;
        else if(precisionStr == "double")
            params.precision = fft_precision_double;
        else
        {
            std::cout << "Invalid precision: " << precisionStr << std::endl;
            return EXIT_FAILURE;
        }
    }

    if(vm.count("transformType"))
        params.transform_type = (fft_transform_type)transform_type_int;
    if(vm.count("itype"))
        params.itype = (fft_array_type)itype_int;
    if(vm.count("otype"))
        params.otype = (fft_array_type)otype_int;

    if(vm.count("help"))
    {
        std::cout << opdesc << std::endl;
        return EXIT_SUCCESS;
    }

    if(vm.count("version"))
    {
        char v[256];
        rocfft_get_version_string(v, 256);
        std::cout << "version " << v << std::endl;
        return EXIT_SUCCESS;
    }

    if(vm.count("ntrial"))
        std::cout << "Running profile with " << ntrial << " samples\n";

    if(!vm.count("length"))
    {
        std::cout << "Please specify transform length!" << std::endl;
        std::cout << opdesc << std::endl;
        return EXIT_SUCCESS;
    }
    parse_arg_ints(lengthArgStr, params.length);
    std::cout << "length:";
    for(auto& i : params.length)
        std::cout << " " << i;
    std::cout << "\n";

    params.placement = vm.count("notInPlace") ? fft_placement_notinplace : fft_placement_inplace;

    if(vm.count("notInPlace"))
        std::cout << "out-of-place\n";
    else
        std::cout << "in-place\n";

    if(vm.count("istride"))
    {
        std::cout << "istride:";
        for(auto& i : params.istride)
            std::cout << " " << i;
        std::cout << "\n";
    }
    if(vm.count("ostride"))
    {
        std::cout << "ostride:";
        for(auto& i : params.ostride)
            std::cout << " " << i;
        std::cout << "\n";
    }

    if(params.idist > 0)
        std::cout << "idist: " << params.idist << "\n";
    if(params.odist > 0)
        std::cout << "odist: " << params.odist << "\n";

    if(vm.count("ioffset"))
    {
        std::cout << "ioffset:";
        for(auto& i : params.ioffset)
            std::cout << " " << i;
        std::cout << "\n";
    }
    if(vm.count("ooffset"))
    {
        std::cout << "ooffset:";
        for(auto& i : params.ooffset)
            std::cout << " " << i;
        std::cout << "\n";
    }

    std::cout << std::flush;

    return offline_tune_problems(params, verbose, ntrial);
}
