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
#include "generator.butterfly.hpp"
#include "generator.pass.hpp"
#include "generator.kernel.hpp"
#include "../../include/radix_table.h"

using namespace StockhamGenerator;

#define large1DThreshold 4096
 
    /* =====================================================================
                Initial parameter used to generate kernels
    =================================================================== */



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

    for(int i=0; i<fft_N.size(); i++)
    {
        params.fft_N[i] = fft_N[i];
    }

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
        DetermineSizes(params.fft_N[0], wgs, nt);//TODO
    }

    assert((nt * params.fft_N[0]) >= wgs);
    assert((nt * params.fft_N[0])%wgs == 0);

    params.fft_numTrans = nt;
    params.fft_workGroupSize = wgs;


    return rocfft_status_success;
}

/* =====================================================================
    WRITE GPU KERNEL FUNCTIONS
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

/* =====================================================================
   WRITE BUTTERFLY DEVICE FUNCTIONS
=================================================================== */
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
   WRITE CPU FUNCTIONS LAUNCHING KERNEL HEADER FILE
=================================================================== */


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
   WRITE CPU FUNCTIONS LAUNCHING KERNEL CPP FILE
=================================================================== */


void WriteCPUWrappersSingle(std::vector<size_t> support_list)
{

    std::string str;

    str += "\n";
    str += "#include \"kernel_launch.h\" \n";
    str += "\n";
    for(size_t i=0;i<support_list.size();i++){

        std::string str_len = std::to_string(support_list[i]);

        str += "#include \"rocfft_kernel_" + str_len + ".h\" \n";
    }

    str += "\n";

    str += "//single precision \n";
    for(size_t i=0;i<support_list.size();i++){

        std::string str_len = std::to_string(support_list[i]);
        str += "POWX_SINGLE_SMALL_GENERATOR( rocfft_internal_dfn_sp_ci_ci_stoc_1_" + str_len +
               ", fft_fwd_ip_len" + str_len + ", fft_back_ip_len" + str_len + ", fft_fwd_op_len" + str_len + ", fft_back_op_len" + str_len + ")\n";
    }


    std::ofstream file;
    std::string fileName = "kernel_launch_single.cpp.h";
    file.open ( fileName );

    if(!file.is_open())
    {
        std::cout << "File: " << fileName << " could not be opened, exiting ...." << std::endl;
    }
    file << str;
    file.close();
}

void WriteCPUWrappersDouble(std::vector<size_t> support_list)
{

    std::string str;

    str += "\n";
    str += "#include \"kernel_launch.h\" \n";
    str += "\n";
    for(size_t i=0;i<support_list.size();i++){

        std::string str_len = std::to_string(support_list[i]);

        str += "#include \"rocfft_kernel_" + str_len + ".h\" \n";

    }

    str += "\n";
    str += "//double precision \n";
    for(size_t i=0;i<support_list.size();i++){

        std::string str_len = std::to_string(support_list[i]);
        str += "POWX_DOUBLE_SMALL_GENERATOR( rocfft_internal_dfn_dp_ci_ci_stoc_1_" + str_len +
               ", fft_fwd_ip_len" + str_len + ", fft_back_ip_len" + str_len + ", fft_fwd_op_len" + str_len + ", fft_back_op_len" + str_len + ")\n";
    }


    std::ofstream file;
    std::string fileName = "kernel_launch_double.cpp.h";
    file.open ( fileName );

    if(!file.is_open())
    {
        std::cout << "File: " << fileName << " could not be opened, exiting ...." << std::endl;
    }
    file << str;
    file.close();
}

/* =====================================================================
   WRITE CPU FUNCTIONS Into Hash Map
=================================================================== */


void WriteCPUFunctionPool(std::vector<size_t> support_list)
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


// *****************************************************
// *****************************************************

int generate_kernel(int len)
{

    std::string programCode;
    rocfft_precision pr = rocfft_precision_single;
    FFTKernelGenKeyParams params;
    BlockComputeType blockComputeType;

    if(len > large1DThreshold) // must decompose into two kernels to do the transform : e.g 8192 = 64*128
    {
        //break into 64*X 

        //length of the FFT in each dimension, <= 3

        size_t large1D_first_dim = 64;
        size_t large1D_second_dim;
        bool blockCompute = false;                
        params.fft_3StepTwiddle = true;
        std::vector<size_t> fft_N(2);

        switch(len)
        {
                case 8192: 
                    {large1D_first_dim = 64; blockCompute = true; }break;
                case 16384: 
                    large1D_first_dim = 64; blockCompute = true; break;
                case 32768: 
                    large1D_first_dim = 128; blockCompute = true; break;
                case 65536: 
                    large1D_first_dim = 256; blockCompute = true; break;
                default: 
                    large1D_first_dim = 64; blockCompute = false;
        }

        for(int i=0;i<2;i++)//break into 2 kernels, that is why the loop ends with 2. 
        {
            if(i==0){
                fft_N[0] = large1D_first_dim;
                fft_N[1] = len/large1D_first_dim;
                blockComputeType = BCT_C2C; //from global memory point view
            }
            else{
                fft_N[1] = large1D_first_dim;
                fft_N[0] = len/large1D_first_dim;
                blockComputeType = BCT_R2C;
            }
                    
            initParams(params, fft_N, blockCompute, blockComputeType);// the last is about set blockCompute or not 
              
            switch(pr)
            {
                        case rocfft_precision_single:
                        {
                            Kernel<rocfft_precision_single> kernel(params);
                            kernel.GenerateKernel(programCode);
                        }
                        break;
                        case rocfft_precision_double:
                        {
                            Kernel<rocfft_precision_double> kernel(params);
                            kernel.GenerateKernel(programCode);
                        }
                        break;
            }
        }
        //its two kernels are in one file
        WriteKernelToFile(programCode, std::to_string(len));

    }
    else{//single kernel

        //length of the FFT in each dimension, <= 3
        std::vector<size_t> fft_N(1); 
        fft_N[0] = len;
        initParams(params, fft_N, false, BCT_C2C);// here the C2C is not enabled, as the third parameter is set as false

        switch(pr)
        {
            case rocfft_precision_single:
            {
                Kernel<rocfft_precision_single> kernel(params);
                kernel.GenerateKernel(programCode);
            }
            break;
            case rocfft_precision_double:
            {
                Kernel<rocfft_precision_double> kernel(params);
                kernel.GenerateKernel(programCode);
            }
            break;
        }

        WriteKernelToFile(programCode, std::to_string(len));

    }

    return 0;
}

// *****************************************************
// *****************************************************

int all_possible(std::vector<size_t> &support_list, size_t i_upper_bound, size_t j_upper_bound, size_t k_upper_bound)
{
    int counter=0;
    size_t upper_bound = std::max(std::max(i_upper_bound, j_upper_bound), k_upper_bound);
    for(size_t i=1;i<=i_upper_bound;i*=5){
        for(size_t j=1;j<=j_upper_bound;j*=3){
            for(size_t k=1;k<=k_upper_bound;k*=2){
                {
                    if( i*j*k <= upper_bound ) {
                        counter++;
                        //printf("Item %d: %d * %d * %d  = %d is below %d \n", (int)counter, (int)i, (int)j, (int)k, i*j*k, upper_bound);
                        size_t len = i*j *k ;
                        support_list.push_back(len);
                    }
                }
            }
        }
    }

    //printf("Total, there are %d valid combinations\n", counter);
    return 0;
}


int main(int argc, char *argv[])
{

    std::string str;
/*
    size_t rad = 10;
    for (size_t d = 0; d<2; d++)
    {
        bool fwd = d ? false : true;
        Butterfly<rocfft_precision_single> bfly1(rad, 1, fwd, true); bfly1.GenerateButterfly(str); str += "\n"; //TODO, does not work for 4, single or double precsion does not matter here.
    }
    printf("Generating rad %d butterfly \n", (int)rad);
    WriteButterflyToFile(str, rad);
    printf("===========================================================================\n");

*/

    std::vector<size_t> support_list;

    if(argc > 1){
        if(strcmp(argv[1], "pow2") == 0){
            //printf("Generating len pow2 FFT kernels\n");
            all_possible(support_list, 1, 1, large1DThreshold);
        }
        else if(strcmp(argv[1], "pow3") == 0){
            //printf("Generating len pow3 FFT kernels\n");
            all_possible(support_list, 1, 2187, 1);
        }
        else if(strcmp(argv[1], "pow5") == 0){
            //printf("Generating len pow5 FFT kernels\n");
            all_possible(support_list, 3125, 1, 1);
        }
        else if(strcmp(argv[1], "pow2,3") == 0){
            //printf("Generating len pow2 and pow3 FFT kernels\n");
            all_possible(support_list, 1, 2187, large1DThreshold);
        }
        else if(strcmp(argv[1], "pow2,5") == 0){
            //printf("Generating len pow2 and pow5 FFT kernels\n");
            all_possible(support_list, 3125, 1, large1DThreshold);
        }
        else if(strcmp(argv[1], "pow3,5") == 0){
            //printf("Generating len pow3 and pow5 FFT kernels\n");
            all_possible(support_list, 3125, 2187, 1);
        }
        else if(strcmp(argv[1], "all") == 0){
            //printf("Generating len mix of 2,3,5 FFT kernels\n");
            all_possible(support_list, 3125, 2187, large1DThreshold);
        }
    }
    else{//if no arguments, generate all possible sizes
         //printf("Generating len mix of 2,3,5 FFT kernels\n");
         all_possible(support_list, 3125, 2187, large1DThreshold);
    }


    //printf("Generating CPU wrappers CPP files \n");
    WriteCPUWrappersSingle(support_list);//TODO: add large1D kernels into CPU functions, need to change CPU side code
    WriteCPUWrappersDouble(support_list);//TODO: add large1D kernels into CPU functions, need to change CPU side code

   //manually add 8K-64K of pow2
    /*support_list.push_back(8192);
    support_list.push_back(16384);                    
    support_list.push_back(32768);
    support_list.push_back(65536);*/

    for(size_t i=0;i<support_list.size();i++){
        //printf("Generating len %d FFT kernels\n", support_list[i]);
        generate_kernel(support_list[i]);
    }
/*
    for(size_t i=7;i<=2401;i*=7){
        printf("Generating len %d FFT kernels\n", (int)i);
        generate_kernel(i);
        support_list.push_back(i);
    }
*/

    //printf("Generating CPU Header \n");
    WriteCPUHeaders(support_list);

    //printf("Generating CPU function into Hash Map \n");
    WriteCPUFunctionPool(support_list);

}
