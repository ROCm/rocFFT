/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#if defined(__NVCC__)
#include "helper_math.h"
#endif

#include "logging.h"
#include "rocfft.h"
#include "rocfft_hip.h"

/*******************************************************************************
 * Static handle data
 ******************************************************************************/
static std::ofstream log_trace_ofs;
static std::ofstream log_bench_ofs;
static std::ofstream log_profile_ofs;

/**
 *  @brief Logging function
 *
 *  @details
 *  open_log_stream Open stream log_os for logging.
 *                  If the environment variable with name
 * environment_variable_name
 *                  is not set, then stream log_os to std::cerr.
 *                  Else open a file at the full logfile path contained in
 *                  the environment variable.
 *                  If opening the file suceeds, stream to the file
 *                  else stream to std::cerr.
 *
 *  @param[in]
 *  environment_variable_name   const char*
 *                              Name of environment variable that contains
 *                              the full logfile path.
 *
 *  @parm[out]
 *  log_os      std::ostream*&
 *              Output stream. Stream to std:cerr if environment_variable_name
 *              is not set, else set to stream to log_ofs
 *
 *  @parm[out]
 *  log_ofs     std::ofstream&
 *              Output file stream. If log_ofs->is_open()==true, then log_os
 *              will stream to log_ofs. Else it will stream to std::cerr.
 */

static void open_log_stream(const char*    environment_variable_name,
                            std::ostream*& log_os,
                            std::ofstream& log_ofs)

{
    // By default, output to cerr
    log_os = &std::cerr;

    // if environment variable is set, open file at logfile_pathname contained in
    // the
    // environment variable
    auto logfile_pathname = getenv(environment_variable_name);
    if(logfile_pathname)
    {
        log_ofs.open(logfile_pathname, std::ios_base::trunc);

        // if log_ofs is open, then stream to log_ofs, else log_os is already set to
        // std::cerr
        if(log_ofs.is_open())
            log_os = &log_ofs;
    }
}

// library setup function, called once in program at the start of library use
rocfft_status rocfft_setup()
{
    // set layer_mode from value of environment variable ROCFFT_LAYER
    auto str_layer_mode = getenv("ROCFFT_LAYER");

    if(str_layer_mode)
    {
        rocfft_layer_mode layer_mode = static_cast<rocfft_layer_mode>(strtol(str_layer_mode, 0, 0));
        LogSingleton::GetInstance().SetLayerMode(layer_mode);

        // open log_trace file
        if(layer_mode & rocfft_layer_mode_log_trace)
            open_log_stream(
                "ROCFFT_LOG_TRACE_PATH", LogSingleton::GetInstance().GetTraceOS(), log_trace_ofs);

        // open log_bench file
        if(layer_mode & rocfft_layer_mode_log_bench)
            open_log_stream(
                "ROCFFT_LOG_BENCH_PATH", LogSingleton::GetInstance().GetBenchOS(), log_bench_ofs);

        // open log_profile file
        if(layer_mode & rocfft_layer_mode_log_profile)
            open_log_stream("ROCFFT_LOG_PROFILE_PATH",
                            LogSingleton::GetInstance().GetProfileOS(),
                            log_profile_ofs);
    }

    log_trace(__func__);
    return rocfft_status_success;
}

// library cleanup function, called once in program after end of library use
rocfft_status rocfft_cleanup()
{
    // Close log files
    if(log_trace_ofs.is_open())
    {
        log_trace_ofs.close();
    }
    if(log_bench_ofs.is_open())
    {
        log_bench_ofs.close();
    }
    if(log_profile_ofs.is_open())
    {
        log_profile_ofs.close();
    }

    log_trace(__func__);
    return rocfft_status_success;
}
