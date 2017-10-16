/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <string.h>
#include <algorithm>
#include "generator.stockham.h"
#include "generator.param.h"
#include "generator.pass.hpp"
#include "generator.kernel.hpp"
#include "../../include/radix_table.h"

using namespace StockhamGenerator;

#define large1DThreshold 4096
 
    /* =====================================================================
                Initial parameter used to generate kernels
    =================================================================== */


extern "C"
rocfft_status initParams (FFTKernelGenKeyParams &params, std::vector<size_t> fft_N,  bool blockCompute, BlockComputeType blockComputeType)
{
            /* =====================================================================
                Parameter : basic plan info
               =================================================================== */


    params.fft_outputLayout = params.fft_inputLayout  = rocfft_array_type_complex_interleaved;

    params.blockCompute = blockCompute;

    params.blockComputeType = blockComputeType;

    bool real_transform = ((params.fft_inputLayout == rocfft_array_type_real) ||
                            (params.fft_outputLayout == rocfft_array_type_real));

            /* =====================================================================
                Parameter : dimension 
               =================================================================== */


    size_t                  batchsize;
    
    params.fft_DataDim = fft_N.size() + 1;

    //TODO: fft_N does not need to know the other dimension
    for(int i=0; i<fft_N.size(); i++)
    {
        params.fft_N[i] = fft_N[i];
    }
    
    params.fft_N[0] = fft_N[0];
            /* =====================================================================
                Parameter: forward, backward scale
               =================================================================== */

    double          forwardScale = 1.0;
    double          backwardScale = 1.0;

    params.fft_fwdScale  = forwardScale;
    params.fft_backScale = backwardScale;


            /* =====================================================================
                Parameter: real FFT
               =================================================================== */

    // Real-Complex simple flag
    // if this is set we do real to-and-from full complex using simple algorithm
    // where imaginary of input is set to zero in forward and imaginary not written in backward
    bool RCsimple = false;
    // Real FFT special flag
    // if this is set it means we are doing the 4th step in the 5-step real FFT breakdown algorithm
    bool realSpecial = false;
    size_t realSpecial_Nr; // this value stores the logical column height (N0) of matrix in the 4th step
                           // length[1] should be 1 + N0/2


    params.fft_RCsimple = RCsimple;
    params.fft_realSpecial = realSpecial;
    //params.fft_realSpecial_Nr = realSpecial_Nr;


    // do twiddle scaling at the beginning pass
    bool    twiddleFront = false;
    params.fft_twiddleFront = twiddleFront;


            /* =====================================================================
                Parameter :  grid and thread blocks (work groups, work items)
                             wgs: work (w) group (g) size (s)
                             nt: number (n) of transforms (t)
                             t_ : temporary
               =================================================================== */

    size_t wgs, nt;
    size_t t_wgs, t_nt;


    KernelCoreSpecs kcs;
    kcs.GetWGSAndNT(params.fft_N[0], t_wgs, t_nt);

    if((t_wgs != 0) && (t_nt != 0) && (MAX_WORK_GROUP_SIZE >= 256))
    {
        wgs = t_wgs;
        nt = t_nt;
    }
    else
    {
       //determine wgs, nt if not in the predefined table
        DetermineSizes(params.fft_N[0], wgs, nt);
    }

    assert((nt * params.fft_N[0]) >= wgs);
    assert((nt * params.fft_N[0])%wgs == 0);

    params.fft_numTrans = nt;
    params.fft_workGroupSize = wgs;


    return rocfft_status_success;
}


/* =====================================================================
   Write butterfly device function to *.h file
=================================================================== */
extern "C" 
void WriteButterflyToFile(std::string &str, int LEN)
{


    std::ofstream file;
    std::string fileName = "rocfft_butterfly_"+std::to_string(LEN)+".h";
    file.open ( fileName );

    if(!file.is_open())
    {
        std::cout << "File: " << fileName << " could not be opened, exiting ...." << std::endl;
    }

    file << str;
    file.close();
}


/* =====================================================================
   Write CPU functions (launching kernel) header to file
=================================================================== */
extern "C" 
void WriteCPUHeaders(std::vector<size_t> support_list)
{

    std::string str;

    str += "\n";
    str += "#pragma once\n";
    str += "#if !defined( kernel_launch_generator_H )\n";
    str += "#define kernel_launch_generator_H \n";

    str += "\n";
    str += "extern \"C\"\n";
    str += "{\n";

    str += "\n";
    str += "//single precision \n";
    for(size_t i=0;i<support_list.size();i++){
        std::string str_len = std::to_string(support_list[i]);
        str += "void rocfft_internal_dfn_sp_ci_ci_stoc_";
        str +=  ( (support_list[i] > large1DThreshold) ? "2_" : "1_" ); 
        str += str_len + "(void *data_p, void *back_p);\n";
    }

    str += "\n";
    str += "//double precision \n";
    for(size_t i=0;i<support_list.size();i++){

        std::string str_len = std::to_string(support_list[i]);
        str += "void rocfft_internal_dfn_dp_ci_ci_stoc_";
        str +=  ( (support_list[i] > large1DThreshold) ? "2_" : "1_" ); 
        str += str_len + "(void *data_p, void *back_p);\n";
    }

    str += "\n";
    str += "}\n";

    str += "\n";
    str += "#endif";

    std::ofstream file;
    std::string fileName = "kernel_launch_generator.h";
    file.open ( fileName );

    if(!file.is_open())
    {
        std::cout << "File: " << fileName << " could not be opened, exiting ...." << std::endl;
    }
    file << str;
    file.close();
}

/* =====================================================================
   Write CPU functions (launching a single kernel) 
   implementation to *.cpp.h file for small sizes
=================================================================== */
extern "C" 
void write_cpu_function_small(std::vector<size_t> support_list, std::string precision)
{
    std::string str;

    std::string large_case_precision = "SINGLE";
    std::string short_name_precision = "sp";

    if(precision == "double"){
        large_case_precision = "DOUBLE";
        short_name_precision = "dp";
    }

    str += "\n";
    str += "#include \"kernel_launch.h\" \n"; //kernel_launch.h has the required macros
    str += "\n";
    for(size_t i=0;i<support_list.size();i++){

        std::string str_len = std::to_string(support_list[i]);

        str += "#include \"rocfft_kernel_" + str_len + ".h\" \n";
    }

    str += "\n";

    str += "//" + precision + " precision \n";
    for(size_t i=0;i<support_list.size();i++){

        std::string str_len = std::to_string(support_list[i]);
        str += "POWX_" + large_case_precision + "_SMALL_GENERATOR( rocfft_internal_dfn_" + short_name_precision + "_ci_ci_stoc_1_" + str_len +
               ", fft_fwd_ip_len" + str_len + ", fft_back_ip_len" + str_len + ", fft_fwd_op_len" + str_len + ", fft_back_op_len" + str_len + ")\n";
    }


    std::ofstream file;
    std::string fileName = "kernel_launch_" + precision + ".cpp.h";
    file.open ( fileName );

    if(!file.is_open())
    {
        std::cout << "File: " << fileName << " could not be opened, exiting ...." << std::endl;
    }
    file << str;
    file.close();
}

/* =====================================================================
   Write CPU functions (launching multiple kernels to finish a transformation) 
   to *.cpp.h file for large sizes 
=================================================================== */

extern "C" 
void write_cpu_function_large(std::vector<size_t> support_list, std::string precision)
{
    // TODO: the first kernel need two twiddles, 

    std::string str;

    std::string large_case_precision = "SINGLE";
    std::string short_name_precision = "sp";

    if(precision == "double"){
        large_case_precision = "DOUBLE";
        short_name_precision = "dp";
    }

    str += "\n";
    str += "#include \"kernel_launch.h\" \n"; //kernel_launch.h has the required macros
    str += "\n";
    for(size_t i=0;i<support_list.size();i++){

        std::string str_len = std::to_string(support_list[i]);

        str += "#include \"rocfft_kernel_" + str_len + ".h\" \n";
    }

    str += "\n";

    str += "//" + precision + " precision \n";

    for(size_t i=0;i<support_list.size();i++){
        std::string str_len = std::to_string(support_list[i]);
        str += "POWX_" + large_case_precision + "_LARGE_GENERATOR( rocfft_internal_dfn_" + short_name_precision + "_ci_ci_stoc_2_" + str_len +
               ", fft_fwd_ip_len" + str_len + ", fft_back_ip_len" + str_len + ", fft_fwd_op_len" + str_len + ", fft_back_op_len" + str_len + ")\n";
    }

    std::ofstream file;
    std::string fileName = "kernel_launch_" + precision + "_large.cpp.h";
    file.open ( fileName );

    if(!file.is_open())
    {
        std::cout << "File: " << fileName << " could not be opened, exiting ...." << std::endl;
    }
    file << str;
    file.close();
}



/* =====================================================================
   Add CPU funtions to function pools (a hash map) 
=================================================================== */
extern "C" 
void AddCPUFunctionToPool(std::vector<size_t> support_list)
{
    std::string str;

    str += "\n";
    str += "#include <iostream> \n";
    str += "#include \"../include/function_pool.h\" \n";
    str += "#include \"kernel_launch_generator.h\" \n";
    str += "\n";
    str += "//build hash map to store the function pointers\n";
    str += "function_pool::function_pool()\n";
    str += "{\n";
    str += "\t//single precision \n";

    for(size_t i=0;i<support_list.size();i++){
        std::string str_len = std::to_string(support_list[i]);
        str += "\tfunction_map_single[" + str_len + "] = &rocfft_internal_dfn_sp_ci_ci_stoc_";
        str +=  ( (support_list[i] > large1DThreshold) ? "2_" : "1_" );
        str += str_len + ";\n";
    }

    str += "\n";
    str += "\t//double precision \n";
    for(size_t i=0;i<support_list.size();i++){
        std::string str_len = std::to_string(support_list[i]);
        str += "\tfunction_map_double[" + str_len + "] = &rocfft_internal_dfn_dp_ci_ci_stoc_";
        str +=  ( (support_list[i] > large1DThreshold) ? "2_" : "1_" );
        str += str_len + ";\n";
    }

    str += "\n";
    str += "}\n";

    std::ofstream file;
    std::string fileName = "function_pool.cpp.h";
    file.open ( fileName );

    if(!file.is_open())
    {
        std::cout << "File: " << fileName << " could not be opened, exiting ...." << std::endl;
    }
    file << str;
    file.close();
}


/* =====================================================================
    Ggenerate the kernels and write to *.h files 
=================================================================== */

void WriteKernelToFile(std::string &str, std::string LEN)
{

    std::ofstream file;
    std::string fileName = "rocfft_kernel_"+LEN+".h";
    file.open ( fileName );

    if(!file.is_open())
    {
        std::cout << "File: " << fileName << " could not be opened, exiting ...." << std::endl;
    }

    file << str;
    file.close();
}

extern "C" 
void generate_kernel_small(size_t len)
{
    std::string programCode;
    FFTKernelGenKeyParams params;

    std::vector<size_t> fft_N(1); 
    fft_N[0] = len;
    initParams(params, fft_N, false, BCT_C2C);// here the C2C is not enabled, as the third parameter is set as false

    Kernel<rocfft_precision_single> kernel(params);//generate data type template kernels regardless of precision 
    kernel.GenerateKernel(programCode);

    WriteKernelToFile(programCode, std::to_string(len));
}

extern "C" 
void generate_kernel_large(std::vector<size_t> large1D_first_dim, std::vector<size_t> large1D_second_dim)
{
    std::string programCode;
    FFTKernelGenKeyParams params;
    BlockComputeType blockComputeType;

    bool isPow2 = true;

    if(isPow2)
    {
        //length of the FFT in each dimension, <= 3
        //generate different combinations, like 8192=64(C2C)*128(R2C). 32768=128(C2C)*256(R2C), notice,128(C2C) != 128(R2C) 
        // the first dim is always type C2C with fft_2StepTwiddle true (1), the second is always R2C with fft_2StepTwiddle false (0)       
        bool blockCompute = true;   //enable blockCompute in large 1D

        std::vector<size_t> fft_N = {1, 1}; 
        //generate C2C type kernels 
        for(int i=0;i<large1D_first_dim.size();i++)
        {                   
            fft_N[0] = large1D_first_dim[i];
            params.fft_3StepTwiddle = true;
            params.name_suffix = "_BCT_C2C";
            initParams(params, fft_N, blockCompute, BCT_C2C);
              
            Kernel<rocfft_precision_single> kernel(params);//generate data type template kernels regardless of precision 
            kernel.GenerateKernel(programCode);

        }

        //generate R2C type kernels 
        for(int i=0;i<large1D_second_dim.size();i++) 
        {                   
            fft_N[0] = large1D_second_dim[i];
            params.fft_3StepTwiddle = false;
            params.name_suffix = "_BCT_R2C";
            initParams(params, fft_N, blockCompute, BCT_R2C);
            
            Kernel<rocfft_precision_single> kernel(params);//generate data type template kernels regardless of precision 
            kernel.GenerateKernel(programCode);  
        }

        WriteKernelToFile(programCode, "large");
    
    }
}



