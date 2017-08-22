/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/



#pragma once
#if !defined( _generator_kernel_H )
#define _generator_kernel_H
#include <stdio.h>
#include <list>
#include "../../include/radix_table.h"
#include "generator.stockham.h"
#include "generator.pass.hpp"
#include "generator.param.h"

 // FFT Stockham Autosort Method
 //
 //   Each pass does one digit reverse in essence. Hence by the time all passes are done, complete
 //   digit reversal is done and output FFT is in correct order. Intermediate FFTs are stored in natural order,
 //   which is not the case with basic Cooley-Tukey algorithm. Natural order in intermediate data makes it
 //   convenient for stitching together passes with different radices.
 //
 //  Basic FFT algorithm:
 //
 //        Pass loop
 //        {
 //            Outer loop
 //            {
 //                Inner loop
 //                {
 //                }
 //            }
 //        }
 //
 //  The sweeps of the outer and inner loop resemble matrix indexing, this matrix changes shape with every pass as noted below
 //
 //   FFT pass diagram (radix 2)
 //
 //                k            k+R                                    k
 //            * * * * * * * * * * * * * * * *                     * * * * * * * *
 //            *   |             |           *                     *   |         *
 //            *   |             |           *                     *   |         *
 //            *   |             |           * LS        -->       *   |         *
 //            *   |             |           *                     *   |         *
 //            *   |             |           *                     *   |         *
 //            * * * * * * * * * * * * * * * *                     *   |         *
 //                         RS                                     *   |         * L
 //                                                                *   |         *
 //                                                                *   |         *
 //                                                                *   |         *
 //                                                                *   |         *
 //                                                                *   |         *
 //                                                                *   |         *
 //                                                                *   |         *
 //                                                                * * * * * * * *
 //                                                                       R
 //
 //
 //    With every pass, the matrix doubles in height and halves in length
 //
 //
 //  N = 2^T = Length of FFT
 //  q = pass loop index
 //  k = outer loop index = (0 ... R-1)
 //  j = inner loop index = (0 ... LS-1)
 //
 //  Tables shows how values change as we go through the passes
 //
 //    q | LS   |  R   |  L  | RS
 //   ___|______|______|_____|___
 //    0 |  1   | N/2  |  2  | N
 //    1 |  2   | N/4  |  4  | N/2
 //    2 |  4   | N/8  |  8  | N/4
 //    . |  .   | .    |  .  | .
 //  T-1 |  N/2 | 1    |  N  | 2
 //
 //
 //   Data Read Order
 //     Radix 2: k*LS + j, (k+R)*LS + j
 //     Radix 3: k*LS + j, (k+R)*LS + j, (k+2R)*LS + j
 //     Radix 4: k*LS + j, (k+R)*LS + j, (k+2R)*LS + j, (k+3R)*LS + j
 //     Radix 5: k*LS + j, (k+R)*LS + j, (k+2R)*LS + j, (k+3R)*LS + j, (k+4R)*LS + j
 //
 //   Data Write Order
 //       Radix 2: k*L + j, k*L + j + LS
 //       Radix 3: k*L + j, k*L + j + LS, k*L + j + 2*LS
 //       Radix 4: k*L + j, k*L + j + LS, k*L + j + 2*LS, k*L + j + 3*LS
 //       Radix 5: k*L + j, k*L + j + LS, k*L + j + 2*LS, k*L + j + 3*LS, k*L + j + 4*LS
 //




namespace StockhamGenerator
{




    class KernelCoreSpecs
    {

        typedef typename std::map<size_t, SpecRecord> SpecTable;
        SpecTable specTable;
        std::vector<SpecRecord> specRecord;
        size_t tableLength;


    public:
        KernelCoreSpecs()
        {
                specRecord = GetRecord();
                //reform an array to a map, the table is store in ../include/radix_table.h
                tableLength = specRecord.size();
                for (size_t i = 0; i<tableLength; i++) specTable[specRecord[i].length] = specRecord[i];

        }

        std::vector<size_t> GetRadices(size_t length)
        {

            std::vector<size_t> radices;

            //printf("tableLength=%d\n", tableLength);
            for(int i=0;i<tableLength;i++){

                if (length == specRecord[i].length){//if find the matched size

                    size_t numPasses = specRecord[i].numPasses;
                    //printf("numPasses=%d, table item %d \n", numPasses, i);
                    for(int j=0;j<numPasses;j++){
                        radices.push_back((specRecord[i].radices)[j]);
                    }
                    break;
                }
            }
            return radices;
        }

        //get working group size and number of transforms
        void GetWGSAndNT(size_t length, size_t &workGroupSize, size_t &numTransforms) const
        {
            workGroupSize = 0;
            numTransforms = 0;

            typename SpecTable::const_iterator it = specTable.find(length);
            if (it != specTable.end())
            {
                workGroupSize = it->second.workGroupSize;
                numTransforms = it->second.numTransforms;
            }
        }
    };


    // FFT kernel generator
    // Kernel calls butterfly and pass
    template <rocfft_precision PR>
    class Kernel
    {
        size_t length;                            // Length of FFT
        size_t workGroupSize;                    // Work group size
        size_t cnPerWI;                            // complex numbers per work-item

        size_t numTrans;                        // The maximum number of FFT-transforms per work-group, internal varaible
        size_t workGroupSizePerTrans;            // Work group subdivision per transform
        size_t numPasses;                        // Number of FFT passes
        std::vector<size_t> radices;            // Base radix at each pass
        std::vector<Pass<PR> > passes;            // Array of pass objects

        bool halfLds;                            // LDS used to store one component (either real or imaginary) at a time
                                                // for passing intermediate data between the passes, if this is set
                                                // then each pass-function should accept same set of registers

        bool linearRegs;                        // scalar registers

                                                // Future optimization ideas
                                                // bool limitRegs;                            // TODO: Incrementally write to LDS, thereby using same set of registers for more than 1 butterflies
                                                // bool combineReadTwMul;                    // TODO: Combine reading into registers and Twiddle multiply

        bool r2c2r;                                // real to complex or complex to real transform
        bool r2c, c2r;
        bool rcFull;
        bool rcSimple;

        bool blockCompute;                        // When we have to compute FFT in blocks (either read or write is along columns)
        size_t blockWidth, blockWGS, blockLDS;

        bool realSpecial;                       // controls related to large1D real FFTs.

        const FFTKernelGenKeyParams params;        // key params


        inline std::string IterRegs(const std::string &pfx, bool initComma = true)
        {
            std::string str = "";

            if (linearRegs)
            {
                if (initComma) str += ", ";

                for (size_t i = 0; i<cnPerWI; i++)
                {
                    if (i != 0) str += ", ";
                    str += pfx; str += "R";
                    str += std::to_string(i);
                }
            }

            return str;
        }

        inline bool IsGroupedReadWritePossible()//TODO
        {
            bool possible = true;
            const size_t *iStride, *oStride;

            if (r2c2r)
                return false;

            if (realSpecial)
                return false;


            iStride = params.fft_inStride;
            oStride = params.fft_outStride;


            for (size_t i = 1; i < params.fft_DataDim; i++)
            {
                if (iStride[i] % 2) { possible = false; break; }
                if (oStride[i] % 2) { possible = false; break; }
            }

            return possible;
        }


        //since it is batching process mutiple matrices by default, calculate the offset block
        inline std::string OffsetCalcBlock(const std::string &offset_name, bool input = true)
        {
            std::string str;

            const size_t *pStride = input ? params.fft_inStride : params.fft_outStride;

            str += "\t"; str += offset_name; str += " = ";
            std::string nextBatch = "batch";
            for (size_t i = (params.fft_DataDim - 1); i>2; i--)
            {
                size_t currentLength = 1;
                for (int j = 2; j<i; j++) currentLength *= params.fft_N[j];
                currentLength *= (params.fft_N[1] / blockWidth);

                str += "("; str += nextBatch; str += "/"; str += std::to_string(currentLength);
                str += ")*"; str += std::to_string(pStride[i]); str += " + ";

                nextBatch = "(" + nextBatch + "%" + std::to_string(currentLength) + ")";
            }

            str += "("; str += nextBatch; str += "/"; str += std::to_string(params.fft_N[1] / blockWidth);
            str += ")*"; str += std::to_string(pStride[2]); str += " + ("; str += nextBatch;
            str += "%"; str += std::to_string(params.fft_N[1] / blockWidth); str += ")*";

            str += std::to_string(blockWidth);
            str += ";\n";

            return str;
        }

        /*OffsetCalc calculates the offset to the memory

          offset_name
             can be ioOffset, iOffset or oOffset, they are size_t type
          stride_name
             can be stride_in or stride_out, they are vector<size_t> type
          output
             if true offset_name2, stride_name2 are enabled
             else not enabled 
        */

        inline std::string OffsetCalc(const std::string offset_name1, const std::string stride_name1, 
                                      const std::string offset_name2, const std::string stride_name2,  
                                      bool output, bool rc_second_index = false)
        {
            std::string str;

            /*===========the comments assume a 16-point FFT============================================*/


            // generate statement like "size_t counter_mod = batch*16 + (me/4);"
            std::string counter_mod;
            if (r2c2r && !rcSimple)
            {
                counter_mod += "(batch*"; counter_mod += std::to_string(2 * numTrans);
                if (rc_second_index) counter_mod += " + 1";
                else                counter_mod += " + 0";

                if (numTrans != 1) { counter_mod += " + 2*(me/"; counter_mod += std::to_string(workGroupSizePerTrans); counter_mod += "))"; }
                else { counter_mod += ")"; }
            }
            else
            {
                if (numTrans == 1) { counter_mod += "batch"; }
                else {
                    counter_mod += "(batch*"; counter_mod += std::to_string(numTrans);
                    counter_mod += " + (me/"; counter_mod += std::to_string(workGroupSizePerTrans); counter_mod += "))";
                }
            }

            str += "\t"; str += "size_t counter_mod = "; str += counter_mod; str += ";\n";

            /*=======================================================*/
            /*  generate a loop like
            if(dim == 1){
                iOffset += counter_mod*strides[1];
            }
            else if(dim == 2){
                int counter_1 = counter_mod / lengths[1];
                int counter_mod_1 = counter_mod % lengths[1];

                iOffset += counter_1*strides[2] + counter_mod_1*strides[1]; 
            }
            else if(dim == 3){
                int counter_2 = counter_mod / (lengths[1] * lengths[2]);
                int counter_mod_2 = counter_mod % (lengths[1] * lengths[2]);

                int counter_1 = counter_mod_2 / lengths[1];
                int counter_mod_1 = counter_mod_2 % lengths[1];

                iOffset += counter_2*strides[3] + counter_1*strides[2] + counter_mod_1*strides[1]; 
            }
            else{
                for(int i = dim; i>1; i--){
                    int currentLength = 1;
                    for(int j=1; j<i; j++){
                        currentLength *= lengths[j];
                    }

                    iOffset += (counter_mod / currentLength)*stride[i];
                    counter_mod = counter_mod % currentLength;
                }
                ioffset += counter_mod*strides[1];
            }
            */

            /*=======================================================*/
            std::string loop;
            loop += "\tif(dim == 1){\n";
            loop += "\t\t" + offset_name1 + " += counter_mod*" + stride_name1 + "[1];\n";
            if (output == true) loop += "\t\t" + offset_name2 + " += counter_mod*" + stride_name2 + "[1];\n";
            loop += "\t}\n";

            loop += "\telse if(dim == 2){\n";
            loop += "\t\tint counter_1 = counter_mod / lengths[1];\n";
            loop += "\t\tint counter_mod_1 = counter_mod % lengths[1];\n";
            loop += "\t\t" + offset_name1 + " += counter_1*"+ stride_name1 + "[2] + counter_mod_1*" + stride_name1 + "[1];\n"; 
            if (output == true) loop += "\t\t" + offset_name2 + " += counter_1*"+ stride_name2 + "[2] + counter_mod_1*" + stride_name2 + "[1];\n"; 
            loop += "\t}\n";

            loop += "\telse if(dim == 3){\n";
            loop += "\t\tint counter_2 = counter_mod / (lengths[1] * lengths[2]);\n";
            loop += "\t\tint counter_mod_2 = counter_mod % (lengths[1] * lengths[2]);\n";
            loop += "\t\tint counter_1 = counter_mod_2 / lengths[1];\n";
            loop += "\t\tint counter_mod_1 = counter_mod_2 % lengths[1];\n";
            loop += "\t\t" + offset_name1 + " += counter_2*"+ stride_name1 + "[3] + counter_1*"+ stride_name1 + "[2] + counter_mod_1*" + stride_name1 + "[1];\n"; 
            if (output == true) 
                loop += "\t\t" + offset_name2 + " += counter_2*"+ stride_name2 + "[3] + counter_1*"+ stride_name2 + "[2] + counter_mod_1*" + stride_name2 + "[1];\n"; 
            loop += "\t}\n";

            loop += "\telse{\n";
            loop += "\t\tfor(int i = dim; i>1; i--){\n";//dim is a runtime variable
            loop += "\t\t\tint currentLength = 1;\n";
            loop += "\t\t\tfor(int j=1; j<i; j++){\n";
            loop += "\t\t\t\tcurrentLength *= lengths[j];\n"; //lengths is a runtime variable
            loop += "\t\t\t}\n";
            loop += "\n";
            loop += "\t\t\t" + offset_name1 + " += (counter_mod / currentLength)*" + stride_name1 + "[i];\n";
            if (output == true)    loop += "\t\t\t" + offset_name2 + " += (counter_mod / currentLength)*" + stride_name2 + "[i];\n";
            loop += "\t\t\tcounter_mod = counter_mod % currentLength;\n";//counter_mod is calculated at runtime
            loop += "\t\t}\n";
            loop += "\t\t" + offset_name1 + "+= counter_mod * " + stride_name1  + "[1];\n";
            if (output == true)    loop += "\t\t" + offset_name2 + "+= counter_mod * " + stride_name2  + "[1];\n";
            loop += "\t}\n";
            
            str += loop;

            return str;
        }

    public:
        Kernel(const FFTKernelGenKeyParams &paramsVal) :
            params(paramsVal), r2c2r(false)

        {
            length = params.fft_N[0];
            workGroupSize = params.fft_workGroupSize;
            numTrans = params.fft_numTrans;


            r2c = false;
            c2r = false;
            // Check if it is R2C or C2R transform
            if (params.fft_inputLayout == rocfft_array_type_real)  r2c = true;
            if (params.fft_outputLayout == rocfft_array_type_real) c2r = true;
            r2c2r = (r2c || c2r);

            if (r2c)
            {
                rcFull = ((params.fft_outputLayout == rocfft_array_type_complex_interleaved) ||
                    (params.fft_outputLayout == rocfft_array_type_complex_planar)) ? true : false;
            }
            if (c2r)
            {
                rcFull = ((params.fft_inputLayout == rocfft_array_type_complex_interleaved) ||
                    (params.fft_inputLayout == rocfft_array_type_complex_planar)) ? true : false;
            }

            rcSimple = params.fft_RCsimple;

            halfLds = true;
            linearRegs = true;

            realSpecial = params.fft_realSpecial;

            blockCompute = params.blockCompute;

            // Make sure we can utilize all Lds if we are going to
            // use blocked columns to compute FFTs
            if (blockCompute)
            {
                assert(length <= 256);  // 256 parameter comes from prototype experiments
                                        // largest length at which block column possible given 32KB LDS limit
                                        // if LDS limit is different this number need to be changed appropriately
                halfLds = false;
                linearRegs = true;
            }

            assert(((length*numTrans) % workGroupSize) == 0);
            cnPerWI = (numTrans * length) / workGroupSize;
            workGroupSizePerTrans = workGroupSize / numTrans;

            // !!!! IMPORTANT !!!! Keep these assertions unchanged, algorithm depend on these to be true
            assert((cnPerWI * workGroupSize) == (numTrans * length));
            assert(cnPerWI <= length); // Don't do more than 1 fft per work-item

                                       // Breakdown into passes

            size_t LS = 1;
            size_t L;
            size_t R = length;
            size_t pid = 0;

            // See if we can get radices from the lookup table, only part of pow2 is in the table
            KernelCoreSpecs kcs;
            std::vector<size_t> radices = kcs.GetRadices(length);
            size_t nPasses = radices.size();
            //params.fft_MaxWorkGroupSize = 512;
            if ((params.fft_MaxWorkGroupSize >= 256) && (nPasses != 0))
            {
                for (size_t i = 0; i<nPasses; i++)
                {
                    size_t rad = radices[i];
                    //printf("length: %d, rad = %d, linearRegs=%d ", (int)length, (int)rad, linearRegs);
                    L = LS * rad;
                    R /= rad;

                    passes.push_back(Pass<PR>(i, length, rad, cnPerWI, L, LS, R, linearRegs, halfLds, r2c, c2r, rcFull, rcSimple, realSpecial));

                    //Pass precallback information to Pass object if its the first pass.
                    //This will be used in single kernel transforms
                    if (params.fft_hasPreCallback && i == 0 && !params.blockCompute)
                    {
                        passes[0].SetPrecallback(params.fft_hasPreCallback);
                    }

                    //Pass post-callback information to Pass object if its the last pass.
                    //This will be used in single kernel transforms
                    if (params.fft_hasPostCallback && i == (nPasses - 1) && !params.blockCompute)
                    {
                        passes[i].SetPostcallback(params.fft_hasPostCallback);
                    }

                    LS *= rad;
                }
                assert(R == 1); // this has to be true for correct radix composition of the length
                numPasses = nPasses;
            }
            else
            {
                //printf("generating radix sequences\n");

                // Possible radices
                size_t cRad[] = { 13,11,10,8,7,6,5,4,3,2,1 }; // Must be in descending order
                size_t cRadSize = (sizeof(cRad) / sizeof(cRad[0]));

                // Generate the radix and pass objects
                while (true)
                {
                    size_t rad;

                    assert(cRadSize >= 1);

                    // Picks the radices in descending order (biggest radix first)
                    for (size_t r = 0; r<cRadSize; r++)
                    {
                        rad = cRad[r];

                        if ((rad > cnPerWI) || (cnPerWI%rad))
                            continue;

                        if (!(R % rad))
                            break;
                    }

                    assert((cnPerWI%rad) == 0);

                    L = LS * rad;
                    R /= rad;

                    radices.push_back(rad);
                    passes.push_back(Pass<PR>(pid, length, rad, cnPerWI, L, LS, R, linearRegs, halfLds, r2c, c2r, rcFull, rcSimple, realSpecial));

                    //Pass precallback information to Pass object if its the first pass.
                    //This will be used in single kernel transforms
                    if (pid == 0 && params.fft_hasPreCallback)
                    {
                        passes[0].SetPrecallback(params.fft_hasPreCallback);
                    }

                    pid++;
                    LS *= rad;

                    assert(R >= 1);
                    if (R == 1)
                        break;
                }// end while
                numPasses = pid;

                //Pass post-callback information to Pass object if its the last pass.
                //This will be used in single kernel transforms
                if (params.fft_hasPostCallback)
                {
                    passes[numPasses - 1].SetPostcallback(params.fft_hasPostCallback);
                }
            }

            assert(numPasses == passes.size());
            assert(numPasses == radices.size());

#ifdef PARMETERS_TO_BE_READ

            ParamRead pr;
            ReadParameterFile(pr);

            radices.clear();
            passes.clear();

            radices = pr.radices;
            numPasses = radices.size();

            LS = 1;
            R = length;
            for (size_t i = 0; i<numPasses; i++)
            {
                size_t rad = radices[i];
                L = LS * rad;
                R /= rad;

                passes.push_back(Pass<PR>(i, length, rad, cnPerWI, L, LS, R, linearRegs));

                LS *= rad;
            }
            assert(R == 1);
#endif

            // Grouping read/writes ok?
            bool grp = IsGroupedReadWritePossible();
            for (size_t i = 0; i < numPasses; i++)
                passes[i].SetGrouping(grp);

            // Store the next pass-object pointers
            if (numPasses > 1)
                for (size_t i = 0; i < (numPasses - 1); i++)
                    passes[i].SetNextPass(&passes[i + 1]);


            if (blockCompute)
            {
                blockWidth = BlockSizes::BlockWidth(length);
                blockWGS = BlockSizes::BlockWorkGroupSize(length);
                blockLDS = BlockSizes::BlockLdsSize(length);
            }
            else
            {
                blockWidth = blockWGS = blockLDS = 0;
            }
        }// end of if ((params.fft_MaxWorkGroupSize >= 256) && (nPasses != 0))

        class BlockSizes
        {
        public:
            enum ValType
            {
                BS_VT_WGS,
                BS_VT_BWD,
                BS_VT_LDS,
            };

            static size_t BlockLdsSize(size_t N) { return GetValue(N, BS_VT_LDS); }
            static size_t BlockWidth(size_t N) { return GetValue(N, BS_VT_BWD); }
            static size_t BlockWorkGroupSize(size_t N) { return GetValue(N, BS_VT_WGS); }

        private:

            static size_t GetValue(size_t N, ValType vt)
            {
                size_t wgs; // preferred work group size
                size_t bwd; // block width to be used
                size_t lds; // LDS size to be used for the block


                KernelCoreSpecs kcs;
                size_t t_wgs, t_nt;
                kcs.GetWGSAndNT(N, t_wgs, t_nt);

                switch (N)
                {
                case 256:    bwd = 8 ;   wgs = (bwd > t_nt) ? 256 : t_wgs; break;
                case 128:    bwd = 8 ;   wgs = (bwd > t_nt) ? 128 : t_wgs; break;
                case 64:    bwd = 16 ;  wgs = (bwd > t_nt) ? 128 : t_wgs; break;
                case 32:    bwd = 32 ;  wgs = (bwd > t_nt) ? 64 : t_wgs; break;
                case 16:    bwd = 64 ;  wgs = (bwd > t_nt) ? 64 : t_wgs; break;
                case 8:        bwd = 128 ; wgs = (bwd > t_nt) ? 64 : t_wgs; break;
                default:    assert(false);
                }

                // block width cannot be less than numTrans, math in other parts of code depend on this assumption
                assert(bwd >= t_nt);

                lds = N*bwd;

                switch (vt)
                {
                case BS_VT_WGS: return wgs;
                case BS_VT_BWD: return bwd;
                case BS_VT_LDS: return lds;
                default: assert(false); return 0;
                }
            }
        };


            /* =====================================================================
                In this GenerateKernel function
                Real2Complex Complex2Real features are not available
                Callback features are not available
               =================================================================== */

        void GenerateKernel(std::string &str)
        {
            // Base type
            std::string rType = RegBaseType<PR>(1);
            // Vector type
            std::string r2Type = RegBaseType<PR>(2);

            bool inInterleaved;     // Input is interleaved format
            bool outInterleaved; // Output is interleaved format
            inInterleaved  = (    (params.fft_inputLayout == rocfft_array_type_complex_interleaved) ||
                                (params.fft_inputLayout == rocfft_array_type_hermitian_interleaved) ) ? true : false;
            outInterleaved = (    (params.fft_outputLayout == rocfft_array_type_complex_interleaved) ||
                                (params.fft_outputLayout == rocfft_array_type_hermitian_interleaved) ) ? true : false;

            // use interleaved LDS when halfLds constraint absent
            bool ldsInterleaved = inInterleaved || outInterleaved;
            ldsInterleaved = halfLds ? false : ldsInterleaved;
            ldsInterleaved = blockCompute ? true : ldsInterleaved;

            bool inReal;  // Input is real format
            bool outReal; // Output is real format
            inReal  = (params.fft_inputLayout == rocfft_array_type_real) ? true : false;
            outReal = (params.fft_outputLayout == rocfft_array_type_real) ? true : false;

            size_t large1D = 0;

            large1D = params.fft_N[0] * params.fft_N[1];

            str += "#include \"kernels/common.h\"\n";
            str += "#include \"rocfft_butterfly_template.h\"\n\n";


            std::string sfx = FloatSuffix<PR>();


            bool cReg = linearRegs ? true : false;
            //printf("cReg is %d \n", cReg);

            // Generate butterflies for all unique radices
            std::list<size_t> uradices;
            for (std::vector<size_t>::const_iterator r = radices.begin(); r != radices.end(); r++)
                uradices.push_back(*r);

            uradices.sort();
            uradices.unique();
            typename std::vector< Pass<PR> >::const_iterator p;


            /* =====================================================================
                write pass functions
                passes call butterfly device functions
                passes use twiddles
                inplace outof place shared the same pass functions
               =================================================================== */

            for (size_t d = 0; d<2; d++)
            {
                bool fwd;
                fwd = d ? false : true;

                double scale = fwd ? params.fft_fwdScale : params.fft_backScale;

                for (p = passes.begin(); p != passes.end(); p++)
                {
                    double s = 1.0;
                    size_t ins = 1, outs = 1;//unit_stride
                    bool gIn = false, gOut = false;
                    bool inIlvd = false, outIlvd = false;
                    bool inRl = false, outRl = false;
                    bool tw3Step = false;


                    if (p == passes.begin() && params.fft_twiddleFront) { tw3Step = params.fft_3StepTwiddle; }
                    if ((p + 1) == passes.end()) { s = scale; if (!params.fft_twiddleFront) tw3Step = params.fft_3StepTwiddle; }


                    if (p == passes.begin()) { inIlvd = inInterleaved;  inRl = inReal;  gIn = true; ins = -1; }
                    // ins = -1 is for non-unit stride, the 1st pass may read strided memory, while the middle pass read/write LDS which guarantees unit-stride
                    if ((p + 1) == passes.end()) { outIlvd = outInterleaved; outRl = outReal; gOut = true; outs = -1; } //-1 is non-unit stride
                    // ins = -1 is for non-unit stride, the last pass may write strided memory
                    if (p != passes.begin()) { inIlvd = ldsInterleaved; }
                    if ((p + 1) != passes.end()) { outIlvd = ldsInterleaved; }

                    p->GeneratePass(fwd, str, tw3Step, params.fft_twiddleFront, inIlvd, outIlvd, inRl, outRl, ins, outs, s, gIn, gOut);
                }

            }


            /* =====================================================================
                Generate Main kernels: call passes
                Generate forward (fwd) cases and backward kernels
                Generate inplace and outof place kernels
               =================================================================== */

                /*  generate fwd or back ward length-point FFT device functions, assuming shared memory (LDS), offset are already pre-calculated\n" */

                for (size_t d = 0; d<2; d++)
                {
                    bool fwd;
                    fwd = d ? false : true;

                    str += "template <typename T, StrideBin sb> \n";
                    str += "__device__ void \n";
                    if(fwd) str += "fwd_len";
                    else  str += "back_len";
                    str += std::to_string(length);
                    str += "_device";
                    str += "(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, "; 
                    str += "unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, real_type_t<T> *lds)\n";
                    str += "{\n";

                    // Setup registers if needed
                    if (linearRegs)
                    {
                        str += "\t"; str += r2Type;
                        str += " "; str += IterRegs("", false);
                        str += ";\n";
                    }

                    if (numPasses == 1)
                    {
                        str += "\t";
                        str += PassName(0, fwd, length);
                        str += "<T, sb>(twiddles, stride_in, stride_out, rw, b, me, 0, 0, lwbIn, lwdOut"; 
                        str += IterRegs("&");
                        str += ");\n";
                    }
                    else
                    {
                        for (typename std::vector<Pass<PR> >::const_iterator p = passes.begin(); p != passes.end(); p++)
                        {
                            std::string exTab = "";

                            str += exTab;
                            str += "\t";
                            str += PassName(p->GetPosition(), fwd, length);
                            str += "<T, sb>(twiddles, stride_in, stride_out, rw, b, me, "; 

                            std::string ldsArgs;
                            if (halfLds) { ldsArgs += "lds, lds"; }
                            else {
                                if (ldsInterleaved) { ldsArgs += "lds"; }
                                else { ldsArgs += "lds, lds + "; ldsArgs += std::to_string(length*numTrans); }
                            }

                            //about offset
                            if (p == passes.begin()) // beginning pass
                            {
                                str += "0, ldsOffset, lwbIn, ";
                                str += ldsArgs; 
                            }
                            else if ((p + 1) == passes.end()) // ending pass
                            {
                                str += "ldsOffset, 0, ";
                                str += ldsArgs; 
                                str += ", lwdOut"; 
                            }
                            else // intermediate pass
                            {
                                str += "ldsOffset, ldsOffset, ";
                                str += ldsArgs; str += ", "; str += ldsArgs; 
                            }

                            str += IterRegs("&"); 
                            str += ");\n";
                            if (!halfLds) { str += exTab; str += "\t__syncthreads();\n"; }
                        }
                    }// if (numPasses == 1)
                    str += "}\n\n";
                }


            for( int place = 0; place<2; place++)
            {
                rocfft_result_placement placeness;
                placeness = place ? rocfft_placement_notinplace : rocfft_placement_inplace;

                for (size_t d = 0; d<2; d++)
                {
                    bool fwd;
                    fwd = d ? false : true;

                    str += "//Kernel configuration: number of threads per thread block: " + std::to_string(workGroupSize) +
                        " transforms: " + std::to_string(numTrans) + " Passes: " + std::to_string(numPasses) + "\n";
                    // FFT kernel begin
                    // Function signature
                    str += "template <typename T, StrideBin sb>\n";
                    str += "__global__ void \n";

                    // kernel name
                    if(fwd) str += "fft_fwd_";
                    else    str += "fft_back_";
                    if(place) str += "op_len";//outof place
                    else    str += "ip_len";//inplace

                    // kernel arguments
                    str += std::to_string(length);
                    str += "( hipLaunchParm lp, ";
                    str += "const " + r2Type + " * __restrict__ twiddles, const size_t dim, const size_t *lengths, ";
                    str += "const size_t *stride_in, ";
                    if(placeness == rocfft_placement_notinplace) str += "const size_t *stride_out, ";
                    str += "const size_t batch_count, ";

                    // Function attributes
                    if (placeness == rocfft_placement_inplace)
                    {


                        assert(inInterleaved == outInterleaved);
                        assert(params.fft_inStride[1] == params.fft_outStride[1]);//here it checks in and outstride match
                        assert(params.fft_inStride[0] == params.fft_outStride[0]);

                        if (inInterleaved)
                        {
                            str += r2Type; str += " * __restrict__   gb";

                            str += ")\n";
                        }
                        else
                        {
                            str += rType; str += " * __restrict__   gbRe, ";
                            str += rType; str += " * __restrict__   gbIm";

                            str += ")\n";
                        }

                    }
                    else
                    {
                        if (inInterleaved)
                        {
                            //str += "const ";
                            str += r2Type; str += " * __restrict__   gbIn, ";//has to remove const qualifier due to HIP on ROCM 1.4
                        }
                        else
                        {
                            str += rType; str += " * __restrict__   gbInRe, ";
                            //str += "const ";
                            str += rType; str += " * __restrict__   gbInIm, ";
                        }

                        if (outInterleaved)
                        {
                              str += r2Type; str += " * __restrict__   gbOut";
                        }
                        else
                        {
                              str += rType; str += " * __restrict__   gbOutRe, ";
                              str += rType; str += " * __restrict__   gbOutIm";
                        }


                        str += ")\n";

                    }

                    str += "{\n";

                    // Initialize
                    str += "\t";
                    str += "unsigned int me = (unsigned int)hipThreadIdx_x;\n\t";
                    str += "unsigned int batch = (unsigned int)hipBlockIdx_x;";
                    str += "\n";


                    // Declare memory pointers
                    str += "\n\t";



                    if (placeness == rocfft_placement_inplace)
                    {
                        str += "unsigned int ioOffset = 0;\n\t";

                        //Skip if callback is set
                        if (!params.fft_hasPreCallback || !params.fft_hasPostCallback)
                        {
                            if (inInterleaved)
                            {
                                  str += r2Type; str += " *lwb;\n";
                            }
                            else
                            {
                                  str += rType; str += " *lwbRe;\n\t";
                                  str += rType; str += " *lwbIm;\n";
                            }
                        }
                        str += "\n";
                    }
                    else
                    {
                        str += "unsigned int iOffset = 0;\n\t";
                        str += "unsigned int oOffset = 0;\n\t";

                        //Skip if precallback is set
                        if (!(params.fft_hasPreCallback))
                        {
                            if (inInterleaved)
                            {
                                  str += r2Type; str += " *lwbIn;\n\t";
                            }
                            else
                            {
                                  str += rType; str += " *lwbInRe;\n\t";
                                  str += rType; str += " *lwbInIm;\n\t";
                            }
                        }

                        //Skip if postcallback is set
                        if (!params.fft_hasPostCallback)
                        {
                            if (outInterleaved)
                            {
                                  str += r2Type; str += " *lwbOut;\n";
                            }
                            else
                            {
                                  str += rType; str += " *lwbOutRe;\n\t";
                                  str += rType; str += " *lwbOutIm;\n";
                            }
                        }
                        str += "\n";
                    }


                    // Conditional read-write ('rw') controls each thread behavior when it is not divisible
                    // for 2D, 3D layout, the "upper_count" viewed by kernels is batch_count*length[1]*length[2]*...*length[dim-1]
                    // because we flatten other dimensions to 1D dimension when configurating the thread blocks  
                    if ((numTrans > 1) && !blockCompute)
                    {
                            str += "\tunsigned int upper_count = batch_count;\n";
                            str += "\tfor(int i=1; i<dim; i++){\n";
                            str += "\t\tupper_count *= lengths[i];\n";
                            str += "\t}\n";
                            str += "\tunsigned int rw = (me < (upper_count ";
                            str += " - batch*"; str += std::to_string(numTrans); str += ")*";
                            str += std::to_string(workGroupSizePerTrans); str += ") ? 1 : 0;\n\n";
                    }
                    else
                    {
                            str += "\tunsigned int rw = 1;\n\n";
                    }

                    //The following lines suppress warning; when rw=1, generator directly puts 1 as the pass device function
                    str += "\t//suppress warning\n";
                    str += "\t#ifdef __NVCC__\n";
                    str += "\t\t(void)(rw == rw);\n";
                    str += "\t#else\n";
                    str += "\t\t(void)rw;\n";
                    str += "\t#endif\n";
                    
                    
                    // Transform index for 3-step twiddles
                    if (params.fft_3StepTwiddle && !blockCompute)
                    {
                        if (numTrans == 1)
                        {
                            str += "\tunsigned int b = batch%";
                        }
                        else
                        {
                            str += "\tunsigned int b = (batch*"; str += std::to_string(numTrans); str += " + (me/";
                            str += std::to_string(workGroupSizePerTrans); str += "))%";
                        }

                        str += std::to_string(params.fft_N[1]); str += ";\n\n";

                    }
                    else
                    {
                        str += "\tunsigned int b = 0;\n\n";
                    }

                // Setup memory pointers


                    if (placeness == rocfft_placement_inplace)
                    {

                        str += OffsetCalc("ioOffset", "stride_in", "", "", false);

                        str += "\t";

                        //Skip if callback is set
                        if (!params.fft_hasPreCallback || !params.fft_hasPostCallback)
                        {
                            if (inInterleaved)
                            {
                                str += "lwb = gb + ioOffset;\n";
                            }
                            else
                            {
                                str += "lwbRe = gbRe + ioOffset;\n\t";
                                str += "lwbIm = gbIm + ioOffset;\n";
                            }
                        }
                        str += "\n";
                    }
                    else
                    {

                        str += OffsetCalc("iOffset", "stride_in", "oOffset", "stride_out", true);


                        str += "\t";

                        //Skip if precallback is set
                        if (!(params.fft_hasPreCallback))
                        {
                            if (inInterleaved)
                            {
                                str += "lwbIn = gbIn + iOffset;\n\t";
                            }
                            else
                            {
                                str += "lwbInRe = gbInRe + iOffset;\n\t";
                                str += "lwbInIm = gbInIm + iOffset;\n\t";
                            }
                        }

                        //Skip if postcallback is set
                        if (!params.fft_hasPostCallback)
                        {
                            if (outInterleaved)
                            {
                                str += "lwbOut = gbOut + oOffset;\n";
                            }
                            else
                            {
                                str += "lwbOutRe = gbOutRe + oOffset;\n\t";
                                str += "lwbOutIm = gbOutIm + oOffset;\n";
                            }
                        }
                        str += "\n";
                    }


                    std::string inOffset;
                    std::string outOffset;
                    if (placeness == rocfft_placement_inplace && !r2c2r)
                    {
                        inOffset += "ioOffset";
                        outOffset += "ioOffset";
                    }
                    else
                    {
                        inOffset += "iOffset";
                        outOffset += "oOffset";
                    }



                    // Set rw and 'me' per transform
                    // rw string also contains 'b'
                    std::string rw, me;

                    if (r2c2r && !rcSimple)    rw = " rw, b, ";
                    else                    rw = ((numTrans > 1) || realSpecial) ? " rw, b, " : " 1, b, ";

                    if (numTrans > 1) { me += "me%"; me += std::to_string(workGroupSizePerTrans); me += ", "; }
                    else { me += "me, "; }



                    // Buffer strings
                    std::string inBuf, outBuf;


                    if (placeness == rocfft_placement_inplace)
                    {
                        if (inInterleaved)
                        {
                            inBuf = params.fft_hasPreCallback ? "gb, " : "lwb, ";
                            outBuf = params.fft_hasPostCallback ? "gb" : "lwb";
                        }
                        else
                        {
                            inBuf = params.fft_hasPreCallback ? "gbRe, gbIm, " : "lwbRe, lwbIm, ";
                            outBuf = params.fft_hasPostCallback ? "gbRe, gbIm" : "lwbRe, lwbIm";
                        }
                    }
                    else
                    {
                        if (inInterleaved)    inBuf = params.fft_hasPreCallback ? "gbIn, " : "lwbIn, ";
                        else                inBuf = params.fft_hasPreCallback ? "gbInRe, gbInIm, " : "lwbInRe, lwbInIm, ";
                        if (outInterleaved)    outBuf = params.fft_hasPostCallback ? "gbOut" : "lwbOut";
                        else                outBuf = params.fft_hasPostCallback ? "gbOutRe, gbOutIm" : "lwbOutRe, lwbOutIm";
                    }


                    /* =====================================================================
                        call passes in the generated kernel
                       =================================================================== */

                    str += "\t// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds \n";
                    str += "\t// rw, b, me% control read/write; then ldsOffset, lwb, lds\n";

                    size_t ldsSize = halfLds ? length*numTrans : 2 * length*numTrans;
                    ldsSize = ldsInterleaved ? ldsSize / 2 : ldsSize;
                    if (numPasses > 1)
                    {
                            str += "\n\t";
                            str +=  "__shared__  "; str += ldsInterleaved ? r2Type : rType; str += " lds[";
                            str += std::to_string(ldsSize); str += "];\n";
                    }

                    std::string ldsOff;
                    if (numTrans > 1)
                    {
                        ldsOff += "(me/"; ldsOff += std::to_string(workGroupSizePerTrans);
                        ldsOff += ")*"; ldsOff += std::to_string(length);
                    }
                    else
                    {
                        ldsOff += "0";
                    }

                    str += "\t";
                    if(fwd) str += "fwd_len";
                    else  str += "back_len";
                    str += std::to_string(length);
                    str += "_device<T, sb>(twiddles, stride_in[0], stride_in[0],";

                    str += rw;
                    str += me;
                    str += ldsOff + ", "; 

                    str += inBuf + outBuf;
                    str += ", lds);\n" ;

                    str += "}\n\n";
                }// end fwd, backward
            }
        }
    };

};

#endif

