/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#pragma once
#if !defined(_generator_pass_H)
#define _generator_pass_H

#include <assert.h>
#include <iostream>
#include <vector>

#include "generator.stockham.h"

namespace StockhamGenerator
{

    // A pass inside an FFT kernel
    // FFT pass generator
    template <rocfft_precision PR>
    class Pass
    {
        size_t position; // Position in the kernel

        size_t algL; // 'L' value from fft algorithm
        size_t algLS; // 'LS' value
        size_t algR; // 'R' value

        size_t length; // Length of FFT
        size_t radix; // Base radix
        size_t cnPerWI; // Complex numbers per work-item

        size_t workGroupSize; // size of the workgroup = (length / cnPerWI)
        // this number is essentially number of work-items needed to compute 1
        // transform
        // this number will be different from the kernel class workGroupSize if there
        // are multiple transforms per workgroup

        size_t numButterfly; // Number of basic FFT butterflies = (cnPerWI / radix)
        size_t numB1, numB2, numB4; // number of different types of butterflies

        bool r2c; // real to complex transform
        bool c2r; // complex to real transform
        bool rcFull;
        bool rcSimple;

        bool realSpecial; // controls related to large1D real FFTs.

        bool      enableGrouping;
        bool      linearRegs; // scalar registers (non-vectorized registers) to be used
        bool      halfLds; // only half the LDS of a complex length need to be used
        Pass<PR>* nextPass;

        bool fft_doPreCallback; // two call back variable
        bool fft_doPostCallback;

        inline void RegBase(size_t regC, std::string& str) const
        {
            str += "B";
            str += std::to_string(regC);
        }

        inline void RegBaseAndCount(size_t num, std::string& str) const
        {
            str += "C";
            str += std::to_string(num);
        }

        inline void RegBaseAndCountAndPos(const std::string& RealImag,
                                          size_t             radPos,
                                          std::string&       str) const
        {
            str += RealImag;
            str += std::to_string(radPos);
        }

        void RegIndex(size_t             regC,
                      size_t             num,
                      const std::string& RealImag,
                      size_t             radPos,
                      std::string&       str) const
        {
            RegBase(regC, str);
            RegBaseAndCount(num, str);
            RegBaseAndCountAndPos(RealImag, radPos, str);
        }

        void DeclareRegs(const std::string& regType,
                         size_t             regC,
                         size_t             numB,
                         std::string&       passStr) const
        {
            std::string regBase;
            RegBase(regC, regBase);

            if(linearRegs)
            {
                assert(regC == 1);
                assert(numB == numButterfly);
            }

            for(size_t i = 0; i < numB; i++)
            {
                passStr += "\n\t";
                passStr += regType;
                passStr += " ";

                std::string regBaseCount = regBase;
                RegBaseAndCount(i, regBaseCount);

                for(size_t r = 0;; r++)
                {
                    if(linearRegs)
                    {
                        std::string regIndex = "R";
                        RegBaseAndCountAndPos("", i * radix + r, regIndex);

                        passStr += regIndex;
                    }
                    else
                    {
                        std::string regRealIndex(regBaseCount), regImagIndex(regBaseCount);

                        RegBaseAndCountAndPos("R", r, regRealIndex); // real
                        RegBaseAndCountAndPos("I", r, regImagIndex); // imaginary

                        passStr += regRealIndex;
                        passStr += ", ";
                        passStr += regImagIndex;
                    }

                    if(r == radix - 1)
                    {
                        passStr += ";";
                        break;
                    }
                    else
                    {
                        passStr += ", ";
                    }
                }
            }
        }

        inline std::string IterRegArgs() const
        {
            std::string str = "";

            if(linearRegs)
            {
                std::string regType = RegBaseType<PR>(2);

                for(size_t i = 0; i < cnPerWI; i++)
                {
                    if(i != 0)
                        str += ", ";
                    str += regType;
                    str += " *R";
                    str += std::to_string(i);
                }
            }

            return str;
        }

#define SR_READ 1
#define SR_TWMUL 2
#define SR_TWMUL_3STEP 3
#define SR_WRITE 4

#define SR_COMP_REAL 0 // real
#define SR_COMP_IMAG 1 // imag
#define SR_COMP_BOTH 2 // real & imag

        // SweepRegs is to iterate through the registers to do the three basic
        // operations:
        // reading, twiddle multiplication, writing
        void SweepRegs(size_t             flag,
                       bool               fwd,
                       bool               interleaved,
                       size_t             stride,
                       size_t             component,
                       double             scale,
                       bool               frontTwiddle,
                       const std::string& bufferRe,
                       const std::string& bufferIm,
                       const std::string& offset,
                       size_t             regC,
                       size_t             numB,
                       size_t             numPrev,
                       std::string&       passStr,
                       bool               isPrecallVector = false,
                       bool               oddt            = false) const
        {
            assert((flag == SR_READ) || (flag == SR_TWMUL) || (flag == SR_TWMUL_3STEP)
                   || (flag == SR_WRITE));

            const std::string twTable     = TwTableName();
            const std::string tw3StepFunc = TwTableLargeFunc();

            // component: 0 - real, 1 - imaginary, 2 - both
            size_t cStart, cEnd;
            switch(component)
            {
            case SR_COMP_REAL:
                cStart = 0;
                cEnd   = 1;
                break;
            case SR_COMP_IMAG:
                cStart = 1;
                cEnd   = 2;
                break;
            case SR_COMP_BOTH:
                cStart = 0;
                cEnd   = 2;
                break;
            default:
                assert(false);
            }

            // Read/Write logic:
            // The double loop inside pass loop of FFT algorithm is mapped into the
            // workGroupSize work items with each work item handling cnPerWI numbers

            // Read logic:
            // Reads for any pass appear the same with the stockham algorithm when
            // mapped to
            // the work items. The buffer is divided into (L/radix) sized blocks and the
            // values are read in linear order inside each block.

            // Vector reads are possible if we have unit strides
            // since read pattern remains the same for all passes and they are
            // contiguous
            // Writes are not contiguous

            // TODO : twiddle multiplies can be combined with read
            // TODO : twiddle factors can be reordered in the table to do vector reads
            // of them

            // Write logic:
            // outer loop index k and the inner loop index j map to 'me' as follows:
            // In one work-item (1 'me'), there are 'numButterfly' fft butterflies. They
            // are indexed as numButterfly*me + butterflyIndex, where butterflyIndex's
            // range is
            // 0 ... numButterfly-1. The total number of butterflies needed is covered
            // over all
            // the work-items. So essentially the double loop k,j is flattened to fit
            // this linearly
            // increasing 'me'.
            // j = (numButterfly*me + butterflyIndex)%LS
            // k = (numButterfly*me + butterflyIndex)/LS

            std::string twType = RegBaseType<PR>(2);
            std::string rType  = RegBaseType<PR>(1);

            size_t      butterflyIndex = numPrev;
            std::string bufOffset;

            std::string regBase;
            RegBase(regC, regBase);

            bool vector4_write = false;
            // special write back to global memory with float4 grouping, writing 2
            // complex numbers at once
            if(numB && (numB % 2 == 0) && (regC == 1) && (numButterfly % 2 == 0) && (algLS % 2 == 0)
               && (flag == SR_WRITE) && (nextPass == NULL) && interleaved
               && (component == SR_COMP_BOTH) && linearRegs && enableGrouping
               && !fft_doPostCallback)
            {
                assert((numButterfly * workGroupSize) == algLS);
                assert(bufferRe.compare(bufferIm) == 0); // Make sure Real & Imag buffer
                // strings are same for
                // interleaved data

                passStr += "\n\t //Optimization: coalescing into float4/double4 write";
                passStr += "\n\tif(sb == SB_UNIT) {";
                passStr += "\n\t";
                passStr += RegBaseType<PR>(4);
                passStr += " *buff4g = ";
                passStr += "(" + RegBaseType<PR>(4) + "*)" + bufferRe;
                passStr += ";\n\t"; // Assuming 'outOffset' is 0, so not adding it here

                for(size_t r = 0; r < radix;
                    r++) // setting the radix loop outside to facilitate grouped writing
                {
                    butterflyIndex = numPrev;

                    for(size_t i = 0; i < (numB / 2); i++)
                    {
                        std::string regIndexA = "(*R";
                        std::string regIndexB = "(*R";

                        RegBaseAndCountAndPos("", (2 * i + 0) * radix + r, regIndexA);
                        regIndexA += ")";
                        RegBaseAndCountAndPos("", (2 * i + 1) * radix + r, regIndexB);
                        regIndexB += ")";

                        passStr += "\n\t";
                        passStr += "buff4g";
                        passStr += "[ ";
                        passStr += std::to_string(numButterfly / 2);
                        passStr += "*me + ";
                        passStr += std::to_string(butterflyIndex);
                        passStr += " + ";
                        passStr += std::to_string(r * (algLS / 2));
                        passStr += " ]";
                        passStr += " = ";
                        passStr += MakeRegBaseType<PR>(4);
                        passStr += "(";
                        passStr += regIndexA;
                        passStr += ".x, ";
                        passStr += regIndexA;
                        passStr += ".y, ";
                        passStr += regIndexB;
                        passStr += ".x, ";
                        passStr += regIndexB;
                        passStr += ".y) ";
                        if(scale != 1.0f)
                        {
                            passStr += " * ";
                            passStr += FloatToStr(scale);
                            passStr += FloatSuffix<PR>();
                        }
                        passStr += ";";

                        butterflyIndex++;
                    }
                }
                passStr += "\n\t}";
                passStr += "\n\telse{ // such optimization is not possible ";

                vector4_write = true;
            }

            size_t hid         = 0;
            bool   swapElement = false;
            size_t tIter       = numB * radix;

            // block to rearrange reads of adjacent memory locations together
            if(linearRegs && (flag == SR_READ))
            {
                for(size_t r = 0; r < radix; r++)
                {
                    for(size_t i = 0; i < numB; i++)
                    {
                        for(size_t c = cStart; c < cEnd;
                            c++) // component loop: 0 - real, 1 - imaginary
                        {
                            swapElement = (fft_doPreCallback && c2r
                                           && component == SR_COMP_REAL); // reset at start of loop

                            std::string tail;
                            std::string regIndex;
                            std::string regIndexC;
                            regIndex = "(*R";
                            std::string buffer;

                            // Read real & imag at once
                            if(interleaved && (component == SR_COMP_BOTH))
                            {
                                assert(bufferRe.compare(bufferIm) == 0); // Make sure Real & Imag
                                // buffer strings are
                                // same for interleaved
                                // data
                                buffer = bufferRe;
                                RegBaseAndCountAndPos("", i * radix + r, regIndex);
                                regIndex += ")";
                                tail = ";";
                            }
                            else
                            {
                                if(c == 0)
                                {
                                    RegBaseAndCountAndPos("", i * radix + r, regIndex);

                                    hid         = (i * radix + r) / (tIter > 1 ? (tIter / 2) : 1);
                                    swapElement = swapElement && hid != 0;
                                    swapElement = (oddt && ((i * radix + r) >= (tIter - 1)))
                                                      ? false
                                                      : swapElement; // for c2r odd size don't swap
                                    // for last register
                                    if(swapElement)
                                    {
                                        regIndexC = regIndex;
                                        regIndexC += ").y";
                                    }

                                    regIndex += ").x";
                                    buffer = bufferRe;
                                    tail   = interleaved ? ".x;" : ";";
                                }
                                else
                                {
                                    RegBaseAndCountAndPos("", i * radix + r, regIndex);
                                    regIndex += ").y";
                                    buffer = bufferIm;
                                    tail   = interleaved ? ".y;" : ";";
                                }
                            }

                            // get offset
                            bufOffset.clear();
                            bufOffset += offset;
                            bufOffset += " + ( ";
                            bufOffset += std::to_string(numPrev);
                            bufOffset += " + ";
                            bufOffset += "me*";
                            bufOffset += std::to_string(numButterfly);
                            bufOffset += " + ";
                            bufOffset += std::to_string(i);
                            bufOffset += " + ";
                            bufOffset += std::to_string(r * length / radix);
                            bufOffset += " )";
                            bufOffset += (stride == 1) ? " " : "*stride_in";

                            if(swapElement)
                            {
                                passStr += "\n\t";
                                passStr += regIndexC;
                                passStr += " = ";
                                passStr += regIndex;
                                passStr += ";";
                            }

                            passStr += "\n\t";
                            passStr += regIndex;
                            passStr += " = ";

                            passStr += buffer;
                            passStr += "[";
                            passStr += bufOffset;
                            passStr += "]";
                            passStr += tail;

                            // Since we read real & imag at once, we break the loop
                            if(interleaved && (component == SR_COMP_BOTH))
                                break;
                        }
                    }
                }
                return;
            }

            // block to rearrange writes of adjacent memory locations together
            if(linearRegs && (flag == SR_WRITE) && (nextPass == NULL))
            {
                for(size_t r = 0; r < radix; r++)
                {
                    butterflyIndex = numPrev;

                    for(size_t i = 0; i < numB; i++)
                    {
                        if(realSpecial && (nextPass == NULL) && (r > (radix / 2)))
                            break;

                        if(realSpecial && (nextPass == NULL) && (r == radix / 2) && (i != 0))
                            break;

                        if(realSpecial && (nextPass == NULL) && (r == radix / 2) && (i == 0))
                            passStr += "\n\t}\n\tif( rw && !me)\n\t{";

                        std::string regIndexC0;
                        for(size_t c = cStart; c < cEnd;
                            c++) // component loop: 0 - real, 1 - imaginary
                        {
                            std::string tail;
                            std::string regIndex;
                            regIndex = "(*R";
                            std::string buffer;

                            // Write real & imag at once
                            if(interleaved && (component == SR_COMP_BOTH))
                            {
                                assert(bufferRe.compare(bufferIm) == 0); // Make sure Real & Imag
                                // buffer strings are
                                // same for interleaved
                                // data
                                buffer = bufferRe;
                                RegBaseAndCountAndPos("", i * radix + r, regIndex);
                                regIndex += ")";
                                tail = "";
                            }
                            else
                            {
                                if(c == 0)
                                {
                                    RegBaseAndCountAndPos("", i * radix + r, regIndex);
                                    regIndex += ").x";
                                    buffer = bufferRe;
                                    tail   = interleaved ? ".x" : "";
                                }
                                else
                                {
                                    RegBaseAndCountAndPos("", i * radix + r, regIndex);
                                    regIndex += ").y";
                                    buffer = bufferIm;
                                    tail   = interleaved ? ".y" : "";
                                }
                            }

                            bufOffset.clear();
                            bufOffset += offset;
                            bufOffset += " + ( ";
                            if((numButterfly * workGroupSize) > algLS)
                            {
                                bufOffset += "((";
                                bufOffset += std::to_string(numButterfly);
                                bufOffset += "*me + ";
                                bufOffset += std::to_string(butterflyIndex);
                                bufOffset += ")/";
                                bufOffset += std::to_string(algLS);
                                bufOffset += ")*";
                                bufOffset += std::to_string(algL);
                                bufOffset += " + (";
                                bufOffset += std::to_string(numButterfly);
                                bufOffset += "*me + ";
                                bufOffset += std::to_string(butterflyIndex);
                                bufOffset += ")%";
                                bufOffset += std::to_string(algLS);
                                bufOffset += " + ";
                            }
                            else
                            {
                                bufOffset += std::to_string(numButterfly);
                                bufOffset += "*me + ";
                                bufOffset += std::to_string(butterflyIndex);
                                bufOffset += " + ";
                            }
                            bufOffset += std::to_string(r * algLS);
                            bufOffset += " )";
                            bufOffset += (stride == 1) ? " " : "*stride_out";

                            if(scale != 1.0f)
                            {
                                regIndex += " * ";
                                regIndex += FloatToStr(scale);
                                regIndex += FloatSuffix<PR>();
                            }
                            if(c == cStart)
                                regIndexC0 = regIndex;

                            passStr += "\n\t";
                            passStr += buffer;
                            passStr += "[";
                            passStr += bufOffset;
                            passStr += "]";
                            passStr += tail;
                            passStr += " = ";
                            passStr += regIndex;
                            passStr += ";";

                            // Since we write real & imag at once, we break the loop
                            if(interleaved && (component == SR_COMP_BOTH))
                                break;
                        }

                        if(realSpecial && (nextPass == NULL) && (r == radix / 2) && (i == 0))
                            passStr += "\n\t}\n\tif(rw)\n\t{";

                        butterflyIndex++;
                    }
                }

                if(vector4_write)
                    passStr += "\n\t}";
                return;
            }

            for(size_t i = 0; i < numB; i++)
            {
                std::string regBaseCount = regBase;
                RegBaseAndCount(i, regBaseCount);

                if(flag == SR_READ) // read operation
                {
                    // the 'r' (radix index) loop is placed outer to the
                    // 'v' (vector index) loop to make possible vectorized reads

                    for(size_t r = 0; r < radix; r++)
                    {
                        for(size_t c = cStart; c < cEnd;
                            c++) // component loop: 0 - real, 1 - imaginary
                        {
                            std::string tail;
                            std::string regIndex;
                            std::string regIndexC;
                            regIndex = linearRegs ? "(*R" : regBaseCount;
                            std::string buffer;

                            // Read real & imag at once
                            if(interleaved && (component == SR_COMP_BOTH) && linearRegs)
                            {
                                assert(bufferRe.compare(bufferIm) == 0); // Make sure Real & Imag
                                // buffer strings are
                                // same for interleaved
                                // data
                                buffer = bufferRe;
                                RegBaseAndCountAndPos("", i * radix + r, regIndex);
                                regIndex += ")";
                                tail = ";";
                            }
                            else
                            {
                                if(c == 0)
                                {
                                    if(linearRegs)
                                    {
                                        RegBaseAndCountAndPos("", i * radix + r, regIndex);

                                        hid = (i * radix + r) / (numB * radix / 2);
                                        if(fft_doPreCallback && c2r && component == SR_COMP_REAL
                                           && hid != 0)
                                        {
                                            regIndexC = regIndex;
                                            regIndexC += ").y";
                                        }
                                        regIndex += ").x";
                                    }
                                    else
                                    {
                                        RegBaseAndCountAndPos("R", r, regIndex);
                                    }
                                    buffer = bufferRe;
                                    tail   = interleaved ? ".x;" : ";";
                                }
                                else
                                {
                                    if(linearRegs)
                                    {
                                        RegBaseAndCountAndPos("", i * radix + r, regIndex);
                                        regIndex += ").y";
                                    }
                                    else
                                    {
                                        RegBaseAndCountAndPos("I", r, regIndex);
                                    }
                                    buffer = bufferIm;
                                    tail   = interleaved ? ".y;" : ";";
                                }
                            }

                            for(size_t v = 0; v < regC; v++) // TODO: vectorize the reads;
                            // instead of reading individually
                            // for consecutive reads of vector
                            // elements
                            {
                                std::string regIndexSub(regIndex);
                                if(regC != 1)
                                {
                                    regIndexSub += ".s";
                                    regIndexSub += std::to_string(v);
                                }

                                // get offset
                                bufOffset.clear();
                                bufOffset += offset;
                                bufOffset += " + ( ";
                                bufOffset += std::to_string(numPrev);
                                bufOffset += " + ";
                                bufOffset += "me*";
                                bufOffset += std::to_string(numButterfly);
                                bufOffset += " + ";
                                bufOffset += std::to_string(i * regC + v);
                                bufOffset += " + ";
                                bufOffset += std::to_string(r * length / radix);
                                bufOffset += " )";
                                bufOffset += (stride == 1) ? " " : "*stride_in";

                                passStr += "\n\t";
                                passStr += regIndexSub;
                                passStr += " = ";

                                passStr += buffer;
                                passStr += "[";
                                passStr += bufOffset;
                                passStr += "]";
                                passStr += tail;
                            }

                            // Since we read real & imag at once, we break the loop
                            if(interleaved && (component == SR_COMP_BOTH) && linearRegs)
                                break;
                        }
                    }
                }
                else if((flag == SR_TWMUL)
                        || (flag == SR_TWMUL_3STEP)) // twiddle multiplies and writes
                // require that 'r' loop be innermost
                {
                    for(size_t v = 0; v < regC; v++)
                    {
                        for(size_t r = 0; r < radix; r++)
                        {

                            std::string regRealIndex, regImagIndex;
                            regRealIndex = linearRegs ? "(*R" : regBaseCount;
                            regImagIndex = linearRegs ? "(*R" : regBaseCount;

                            if(linearRegs)
                            {
                                RegBaseAndCountAndPos("", i * radix + r, regRealIndex);
                                regRealIndex += ").x";
                                RegBaseAndCountAndPos("", i * radix + r, regImagIndex);
                                regImagIndex += ").y";
                            }
                            else
                            {
                                RegBaseAndCountAndPos("R", r, regRealIndex);
                                RegBaseAndCountAndPos("I", r, regImagIndex);
                            }

                            if(regC != 1)
                            {
                                regRealIndex += ".s";
                                regRealIndex += std::to_string(v);
                                regImagIndex += ".s";
                                regImagIndex += std::to_string(v);
                            }

                            if(flag == SR_TWMUL) // twiddle multiply operation
                            {
                                if(r == 0) // no twiddle muls needed
                                    continue;

                                passStr += "\n\t{\n\t\t";
                                passStr += twType;
                                passStr += " W = ";
                                passStr += twTable;
                                passStr += "[";
                                passStr += std::to_string(algLS - 1);
                                passStr += " + ";
                                passStr += std::to_string(radix - 1);
                                passStr += "*((";
                                passStr += std::to_string(numButterfly);
                                passStr += "*me + ";
                                passStr += std::to_string(butterflyIndex);
                                passStr += ")%";
                                passStr += std::to_string(algLS);
                                passStr += ") + ";
                                passStr += std::to_string(r - 1);
                                passStr += "];\n\t\t";
                            }
                            else // 2, 3, or 4-step twiddle
                            {
                                passStr += "\n\t{\n\t\t";
                                passStr += twType;
                                passStr += " W = ";
                                passStr += tw3StepFunc;
                                passStr += "<T>(twiddles_large, ";

                                if(frontTwiddle)
                                {
                                    assert(linearRegs);
                                    passStr += "(";
                                    passStr += "me*";
                                    passStr += std::to_string(numButterfly);
                                    passStr += " + ";
                                    passStr += std::to_string(i);
                                    passStr += " + ";
                                    passStr += std::to_string(r * length / radix);
                                    passStr += ") * b";
                                }
                                else
                                {
                                    passStr += "((";
                                    passStr += std::to_string(numButterfly);
                                    passStr += "*me + ";
                                    passStr += std::to_string(butterflyIndex);
                                    passStr += ")%";
                                    passStr += std::to_string(algLS);
                                    passStr += " + ";
                                    passStr += std::to_string(r * algLS);
                                    passStr += ") * b";
                                }

                                passStr += " );\n\t\t";
                            }

                            passStr += rType;
                            passStr += " TR, TI;\n\t\t";

                            if(realSpecial && (flag == SR_TWMUL_3STEP))
                            {
                                if(fwd)
                                {
                                    passStr += "if(t==0)\n\t\t{\n\t\t";

                                    passStr += "TR = (W.x * ";
                                    passStr += regRealIndex;
                                    passStr += ") - (W.y * ";
                                    passStr += regImagIndex;
                                    passStr += ");\n\t\t";
                                    passStr += "TI = (W.y * ";
                                    passStr += regRealIndex;
                                    passStr += ") + (W.x * ";
                                    passStr += regImagIndex;
                                    passStr += ");\n\t\t";

                                    passStr += "}\n\t\telse\n\t\t{\n\t\t";

                                    passStr += "TR = (W.x * ";
                                    passStr += regRealIndex;
                                    passStr += ") + (W.y * ";
                                    passStr += regImagIndex;
                                    passStr += ");\n\t\t";
                                    passStr += "TI = (W.y * ";
                                    passStr += regRealIndex;
                                    passStr += ") - (W.x * ";
                                    passStr += regImagIndex;
                                    passStr += ");\n\t\t";

                                    passStr += "}\n\t\t";
                                }
                                else
                                {
                                    passStr += "if(t==0)\n\t\t{\n\t\t";

                                    passStr += "TR = (W.x * ";
                                    passStr += regRealIndex;
                                    passStr += ") + (W.y * ";
                                    passStr += regImagIndex;
                                    passStr += ");\n\t\t";
                                    passStr += "TI = (W.y * ";
                                    passStr += regRealIndex;
                                    passStr += ") - (W.x * ";
                                    passStr += regImagIndex;
                                    passStr += ");\n\t\t";

                                    passStr += "}\n\t\telse\n\t\t{\n\t\t";

                                    passStr += "TR = (W.x * ";
                                    passStr += regRealIndex;
                                    passStr += ") - (W.y * ";
                                    passStr += regImagIndex;
                                    passStr += ");\n\t\t";
                                    passStr += "TI = (W.y * ";
                                    passStr += regRealIndex;
                                    passStr += ") + (W.x * ";
                                    passStr += regImagIndex;
                                    passStr += ");\n\t\t";

                                    passStr += "}\n\t\t";
                                }
                            }
                            else
                            {
                                if(fwd)
                                {
                                    passStr += "TR = (W.x * ";
                                    passStr += regRealIndex;
                                    passStr += ") - (W.y * ";
                                    passStr += regImagIndex;
                                    passStr += ");\n\t\t";
                                    passStr += "TI = (W.y * ";
                                    passStr += regRealIndex;
                                    passStr += ") + (W.x * ";
                                    passStr += regImagIndex;
                                    passStr += ");\n\t\t";
                                }
                                else
                                {
                                    passStr += "TR =  (W.x * ";
                                    passStr += regRealIndex;
                                    passStr += ") + (W.y * ";
                                    passStr += regImagIndex;
                                    passStr += ");\n\t\t";
                                    passStr += "TI = -(W.y * ";
                                    passStr += regRealIndex;
                                    passStr += ") + (W.x * ";
                                    passStr += regImagIndex;
                                    passStr += ");\n\t\t";
                                }
                            }

                            passStr += regRealIndex;
                            passStr += " = TR;\n\t\t";
                            passStr += regImagIndex;
                            passStr += " = TI;\n\t}\n";
                        }

                        butterflyIndex++;
                    }
                }
                else // write operation
                {
                    for(size_t v = 0; v < regC; v++)
                    {
                        for(size_t r = 0; r < radix; r++)
                        {
                            if(realSpecial && (nextPass == NULL) && (r > (radix / 2)))
                                break;

                            if(realSpecial && (nextPass == NULL) && (r == radix / 2) && (i != 0))
                                break;

                            if(realSpecial && (nextPass == NULL) && (r == radix / 2) && (i == 0))
                                passStr += "\n\t}\n\tif( rw && !me)\n\t{";

                            std::string regIndexC0;

                            for(size_t c = cStart; c < cEnd;
                                c++) // component loop: 0 - real, 1 - imaginary
                            {
                                std::string tail;
                                std::string regIndex;
                                regIndex = linearRegs ? "(*R" : regBaseCount;
                                std::string buffer;

                                // Write real & imag at once
                                if(interleaved && (component == SR_COMP_BOTH) && linearRegs)
                                {
                                    assert(bufferRe.compare(bufferIm) == 0); // Make sure Real &
                                    // Imag buffer strings
                                    // are same for
                                    // interleaved data
                                    buffer = bufferRe;
                                    RegBaseAndCountAndPos("", i * radix + r, regIndex);
                                    regIndex += ")";
                                    tail = "";
                                }
                                else
                                {
                                    if(c == 0)
                                    {
                                        if(linearRegs)
                                        {
                                            RegBaseAndCountAndPos("", i * radix + r, regIndex);
                                            regIndex += ").x";
                                        }
                                        else
                                        {
                                            RegBaseAndCountAndPos("R", r, regIndex);
                                        }
                                        buffer = bufferRe;
                                        tail   = interleaved ? ".x" : "";
                                    }
                                    else
                                    {
                                        if(linearRegs)
                                        {
                                            RegBaseAndCountAndPos("", i * radix + r, regIndex);
                                            regIndex += ").y";
                                        }
                                        else
                                        {
                                            RegBaseAndCountAndPos("I", r, regIndex);
                                        }
                                        buffer = bufferIm;
                                        tail   = interleaved ? ".y" : "";
                                    }
                                }

                                if(regC != 1)
                                {
                                    regIndex += ".s";
                                    regIndex += std::to_string(v);
                                }

                                passStr += "\n\t";

                                if(scale != 1.0f)
                                {
                                    regIndex += " * ";
                                    regIndex += FloatToStr(scale);
                                    regIndex += FloatSuffix<PR>();
                                }
                                if(c == 0)
                                    regIndexC0 += regIndex;

                                bufOffset.clear();
                                bufOffset += offset;
                                bufOffset += " + ( ";
                                if((numButterfly * workGroupSize) > algLS)
                                {
                                    bufOffset += "((";
                                    bufOffset += std::to_string(numButterfly);
                                    bufOffset += "*me + ";
                                    bufOffset += std::to_string(butterflyIndex);
                                    bufOffset += ")/";
                                    bufOffset += std::to_string(algLS);
                                    bufOffset += ")*";
                                    bufOffset += std::to_string(algL);
                                    bufOffset += " + (";
                                    bufOffset += std::to_string(numButterfly);
                                    bufOffset += "*me + ";
                                    bufOffset += std::to_string(butterflyIndex);
                                    bufOffset += ")%";
                                    bufOffset += std::to_string(algLS);
                                    bufOffset += " + ";
                                }
                                else
                                {
                                    bufOffset += std::to_string(numButterfly);
                                    bufOffset += "*me + ";
                                    bufOffset += std::to_string(butterflyIndex);
                                    bufOffset += " + ";
                                }

                                bufOffset += std::to_string(r * algLS);
                                bufOffset += " )";
                                bufOffset += (stride == 1) ? " " : "*stride_out";

                                passStr += buffer;
                                passStr += "[";
                                passStr += bufOffset;
                                passStr += "]";
                                passStr += tail;
                                passStr += " = ";
                                passStr += regIndex;
                                passStr += ";";

                                // Since we write real & imag at once, we break the loop
                                if(interleaved && (component == SR_COMP_BOTH) && linearRegs)
                                    break;
                            } // end for(size_t c=cStart; c<cEnd; c++)

                            if(realSpecial && (nextPass == NULL) && (r == radix / 2) && (i == 0))
                                passStr += "\n\t}\n\tif(rw)\n\t{";

                        } // end for radix

                        butterflyIndex++;
                    } // end for for(size_t v=0; v<regC; v++)

                } // end else write operation
            }

            assert(butterflyIndex <= numButterfly);
        }

        // Special SweepRegs function to carry out some R-C/C-R specific operations
        void SweepRegsRC(size_t             flag,
                         bool               fwd,
                         bool               interleaved,
                         size_t             stride,
                         size_t             component,
                         double             scale,
                         bool               setZero,
                         bool               batch2,
                         bool               oddt,
                         const std::string& bufferRe,
                         const std::string& bufferIm,
                         const std::string& offset,
                         std::string&       passStr) const
        {
            assert((flag == SR_READ) || (flag == SR_WRITE));

            // component: 0 - real, 1 - imaginary, 2 - both
            size_t cStart, cEnd;
            switch(component)
            {
            case SR_COMP_REAL:
                cStart = 0;
                cEnd   = 1;
                break;
            case SR_COMP_IMAG:
                cStart = 1;
                cEnd   = 2;
                break;
            case SR_COMP_BOTH:
                cStart = 0;
                cEnd   = 2;
                break;
            default:
                assert(false);
            }

            std::string rType = RegBaseType<PR>(1);

            assert(r2c || c2r);
            assert(linearRegs);
            bool singlePass = ((position == 0) && (nextPass == NULL));

            size_t numCR = numButterfly * radix;
            if(!(numCR % 2))
                assert(!oddt);

            size_t rStart = 0;
            size_t rEnd   = numCR;

            bool oddp = ((numCR % 2) && (numCR > 1) && !setZero);
            if(oddp)
            {
                if(oddt)
                {
                    rStart = numCR - 1;
                    rEnd   = numCR + 1;
                }
                else
                {
                    rStart = 0;
                    rEnd   = numCR - 1;
                }
            }

            if(!oddp)
                assert(!oddt);

            for(size_t r = rStart; r < rEnd; r++)
            {
                std::string val1StrExt;

                for(size_t c = cStart; c < cEnd; c++) // component loop: 0 - real, 1 - imaginary
                {
                    if(flag == SR_READ) // read operation
                    {
                        std::string tail, tail2;
                        std::string regIndex = "(*R";
                        std::string buffer;

                        RegBaseAndCountAndPos("", r, regIndex);

                        if(c == 0)
                        {
                            regIndex += ").x";
                            buffer = bufferRe;
                            tail   = interleaved ? ".x;" : ";";
                            tail2  = interleaved ? ".y;" : ";";
                        }
                        else
                        {
                            regIndex += ").y";
                            buffer = bufferIm;
                            tail   = interleaved ? ".y;" : ";";
                            tail2  = interleaved ? ".x;" : ";";
                        }

                        size_t bid = numCR / 2;
                        bid        = bid ? bid : 1;
                        size_t cid, lid;

                        if(oddt)
                        {
                            cid = r % 2;
                            lid = 1 + (numCR / 2);
                        }
                        else
                        {
                            cid = r / bid;
                            lid = 1 + r % bid;
                        }

                        std::string oddpadd = oddp ? " (me/2) + " : " ";

                        std::string idxStr, idxStrRev;
                        if((length <= 2) || ((length & (length - 1)) != 0))
                        {
                            idxStr += std::to_string(bid);
                            idxStr += "*me +";
                            idxStr += oddpadd;
                            idxStr += std::to_string(lid);
                        }
                        else
                        {
                            idxStr += "me + ";
                            idxStr += std::to_string(1 + length * (r % bid) / numCR);
                            idxStr += oddpadd;
                        }
                        idxStrRev += std::to_string(length);
                        idxStrRev += " - (";
                        idxStrRev += idxStr;
                        idxStrRev += " )";

                        bool act = (fwd || ((cid == 0) && (!batch2)) || ((cid != 0) && batch2));
                        if(act)
                        {
                            passStr += "\n\t";
                            passStr += regIndex;
                            passStr += " = ";
                        }

                        if(setZero)
                        {
                            if(act)
                                passStr += "0;";
                        }
                        else
                        {
                            if(act)
                            {

                                passStr += buffer;
                                passStr += "[";

                                passStr += offset;
                                passStr += " + ( ";
                            }

                            if(fwd)
                            {
                                if(cid == 0)
                                    passStr += idxStr;
                                else
                                    passStr += idxStrRev;
                            }
                            else
                            {
                                if(cid == 0)
                                {
                                    if(!batch2)
                                        passStr += idxStr;
                                }
                                else
                                {
                                    if(batch2)
                                        passStr += idxStr;
                                }
                            }

                            if(act)
                            {
                                passStr += " )";
                                passStr += (stride == 1) ? " " : "*stride_in";

                                passStr += "]";

                                if(fwd)
                                {
                                    passStr += tail;
                                }
                                else
                                {
                                    if(!batch2)
                                        passStr += tail;
                                    else
                                        passStr += tail2;
                                }
                            }
                        }
                    }
                    else // write operation
                    {
                        std::string tail;
                        std::string regIndex     = "(*R";
                        std::string regIndexPair = "(*R";
                        std::string buffer;

                        // Write real & imag at once
                        if(interleaved && (component == SR_COMP_BOTH))
                        {
                            assert(bufferRe.compare(bufferIm) == 0); // Make sure Real & Imag
                            // buffer strings are same
                            // for interleaved data
                            buffer = bufferRe;
                        }
                        else
                        {
                            if(c == 0)
                            {
                                buffer = bufferRe;
                                tail   = interleaved ? ".x" : "";
                            }
                            else
                            {
                                buffer = bufferIm;
                                tail   = interleaved ? ".y" : "";
                            }
                        }

                        size_t bid, cid, lid;
                        if(singlePass && fwd)
                        {
                            bid = 1 + radix / 2;
                            lid = r;
                            cid = r / bid;

                            RegBaseAndCountAndPos("", r, regIndex);
                            regIndex += ")";
                            RegBaseAndCountAndPos("", (radix - r) % radix, regIndexPair);
                            regIndexPair += ")";
                        }
                        else
                        {
                            bid = numCR / 2;

                            if(oddt)
                            {
                                cid = r % 2;
                                lid = 1 + (numCR / 2);

                                RegBaseAndCountAndPos("", r, regIndex);
                                regIndex += ")";
                                RegBaseAndCountAndPos("", r + 1, regIndexPair);
                                regIndexPair += ")";
                            }
                            else
                            {
                                cid = r / bid;
                                lid = 1 + r % bid;

                                RegBaseAndCountAndPos("", r, regIndex);
                                regIndex += ")";
                                RegBaseAndCountAndPos("", r + bid, regIndexPair);
                                regIndexPair += ")";
                            }
                        }

                        if(!cid)
                        {
                            std::string oddpadd = oddp ? " (me/2) + " : " ";

                            std::string sclStr = "";
                            if(scale != 1.0f)
                            {
                                sclStr += " * ";
                                sclStr += FloatToStr(scale);
                                sclStr += FloatSuffix<PR>();
                            }

                            if(fwd)
                            {
                                std::string idxStr, idxStrRev;
                                if((length <= 2) || ((length & (length - 1)) != 0))
                                {
                                    idxStr += std::to_string(length / (2 * workGroupSize));
                                    idxStr += "*me +";
                                    idxStr += oddpadd;
                                    idxStr += std::to_string(lid);
                                }
                                else
                                {
                                    idxStr += "me + ";
                                    idxStr += std::to_string(1 + length * (r % bid) / numCR);
                                    idxStr += oddpadd;
                                }
                                idxStrRev += std::to_string(length);
                                idxStrRev += " - (";
                                idxStrRev += idxStr;
                                idxStrRev += " )";

                                std::string val1Str, val2Str;

                                val1Str += "\n\t";
                                val1Str += buffer;
                                val1Str += "[";
                                val1Str += offset;
                                val1Str += " + ( ";
                                val1Str += idxStr;
                                val1Str += " )";
                                val1Str += (stride == 1) ? " " : "*stride_out";
                                val1Str += "]";
                                val1Str += tail;
                                val1Str += " = ";

                                val2Str += "\n\t";
                                val2Str += buffer;
                                val2Str += "[";
                                val2Str += offset;
                                val2Str += " + ( ";
                                val2Str += idxStrRev;
                                val2Str += " )";
                                val1Str += (stride == 1) ? " " : "*stride_out";
                                val2Str += "]";
                                val2Str += tail;
                                val2Str += " = ";

                                std::string real1, imag1, real2, imag2;

                                real1 += "(";
                                real1 += regIndex;
                                real1 += ".x + ";
                                real1 += regIndexPair;
                                real1 += ".x)*0.5";
                                imag1 += "(";
                                imag1 += regIndex;
                                imag1 += ".y - ";
                                imag1 += regIndexPair;
                                imag1 += ".y)*0.5";
                                real2 += "(";
                                real2 += regIndex;
                                real2 += ".y + ";
                                real2 += regIndexPair;
                                real2 += ".y)*0.5";
                                imag2 += "(-";
                                imag2 += regIndex;
                                imag2 += ".x + ";
                                imag2 += regIndexPair;
                                imag2 += ".x)*0.5";

                                if(interleaved && (component == SR_COMP_BOTH))
                                {
                                    val1Str += "(";
                                    val1Str += RegBaseType<PR>(2);
                                    val1Str += ")( ";
                                    val2Str += "(";
                                    val2Str += RegBaseType<PR>(2);
                                    val2Str += ")( ";

                                    if(!batch2)
                                    {
                                        val1Str += real1;
                                        val1Str += ", ";
                                        val1Str += "+";
                                        val1Str += imag1;
                                        val2Str += real1;
                                        val2Str += ", ";
                                        val2Str += "-";
                                        val2Str += imag1;
                                    }
                                    else
                                    {
                                        val1Str += real2;
                                        val1Str += ", ";
                                        val1Str += "+";
                                        val1Str += imag2;
                                        val2Str += real2;
                                        val2Str += ", ";
                                        val2Str += "-";
                                        val2Str += imag2;
                                    }

                                    val1Str += " )";
                                    val2Str += " )";
                                }
                                else
                                {
                                    val1Str += " (";
                                    val2Str += " (";
                                    if(c == 0)
                                    {
                                        if(!batch2)
                                        {
                                            val1Str += real1;
                                            val2Str += real1;
                                        }
                                        else
                                        {
                                            val1Str += real2;
                                            val2Str += real2;
                                        }
                                    }
                                    else
                                    {
                                        if(!batch2)
                                        {
                                            val1Str += "+";
                                            val1Str += imag1;
                                            val2Str += "-";
                                            val2Str += imag1;
                                        }
                                        else
                                        {
                                            val1Str += "+";
                                            val1Str += imag2;
                                            val2Str += "-";
                                            val2Str += imag2;
                                        }
                                    }
                                    val1Str += " )";
                                    val2Str += " )";
                                }

                                val1Str += sclStr;
                                val2Str += sclStr;

                                val1Str += ";";

                                passStr += val1Str;
                                if(rcFull)
                                {
                                    passStr += val2Str;
                                    passStr += ";";
                                }
                            }
                            else
                            {
                                std::string idxStr, idxStrRev;
                                if((length <= 2) || ((length & (length - 1)) != 0))
                                {
                                    idxStr += std::to_string(bid);
                                    idxStr += "*me +";
                                    idxStr += oddpadd;
                                    idxStr += std::to_string(lid);
                                }
                                else
                                {
                                    idxStr += "me + ";
                                    idxStr += std::to_string(1 + length * (r % bid) / numCR);
                                    idxStr += oddpadd;
                                }
                                idxStrRev += std::to_string(length);
                                idxStrRev += " - (";
                                idxStrRev += idxStr;
                                idxStrRev += " )";

                                passStr += "\n\t";
                                passStr += buffer;
                                passStr += "[";
                                passStr += offset;
                                passStr += " + ( ";

                                if(!batch2)
                                    passStr += idxStr;
                                else
                                    passStr += idxStrRev;

                                passStr += " )";
                                passStr += (stride == 1) ? " " : "*stride_out";
                                passStr += "]";
                                passStr += tail;
                                passStr += " = ";

                                passStr += "( ";
                                if(c == 0)
                                {
                                    regIndex += ".x";
                                    regIndexPair += fft_doPreCallback ? ".y" : ".x";

                                    if(!batch2)
                                    {
                                        passStr += regIndex;
                                        passStr += " - ";
                                        passStr += regIndexPair;
                                    }
                                    else
                                    {
                                        passStr += regIndex;
                                        passStr += " + ";
                                        passStr += regIndexPair;
                                    }
                                }
                                else
                                {
                                    regIndex += ".y";
                                    regIndexPair += (fft_doPreCallback && oddt) ? ".x" : ".y";

                                    if(!batch2)
                                    {
                                        passStr += regIndex;
                                        passStr += " + ";
                                        passStr += regIndexPair;
                                    }
                                    else
                                    {
                                        passStr += " - ";
                                        passStr += regIndex;
                                        passStr += " + ";
                                        passStr += regIndexPair;
                                    }
                                }
                                passStr += " )";
                                passStr += sclStr;
                                passStr += ";";
                            }

                            // Since we write real & imag at once, we break the loop
                            if(interleaved && (component == SR_COMP_BOTH))
                                break;
                        }
                    }
                }
            }
        }

        void CallButterfly(const std::string& bflyName,
                           size_t             regC,
                           size_t             numB,
                           std::string&       passStr) const
        {
            std::string regBase;
            RegBase(regC, regBase);

            for(size_t i = 0; i < numB; i++)
            {
                std::string regBaseCount = regBase;
                RegBaseAndCount(i, regBaseCount);

                passStr += "\n\t";
                passStr += bflyName;
                passStr += "(";

                for(size_t r = 0;; r++)
                {
                    if(linearRegs)
                    {
                        std::string regIndex = "R";
                        RegBaseAndCountAndPos("", i * radix + r, regIndex);

                        passStr += regIndex;
                    }
                    else
                    {
                        std::string regRealIndex(regBaseCount);
                        std::string regImagIndex(regBaseCount);
                        RegBaseAndCountAndPos("R", r, regRealIndex);
                        RegBaseAndCountAndPos("I", r, regImagIndex);

                        passStr += "&";
                        passStr += regRealIndex;
                        passStr += ", ";
                        passStr += "&";
                        passStr += regImagIndex;
                    }

                    if(r == radix - 1)
                    {
                        passStr += ");";
                        break;
                    }
                    else
                    {
                        passStr += ", ";
                    }
                }
            }
        }

    public:
        Pass(size_t positionVal,
             size_t lengthVal,
             size_t radixVal,
             size_t cnPerWIVal,
             size_t L,
             size_t LS,
             size_t R,
             bool   linearRegsVal,
             bool   halfLdsVal,
             bool   r2cVal,
             bool   c2rVal,
             bool   rcFullVal,
             bool   rcSimpleVal,
             bool   realSpecialVal)
            : position(positionVal)
            , length(lengthVal)
            , radix(radixVal)
            , cnPerWI(cnPerWIVal)
            , algL(L)
            , algLS(LS)
            , algR(R)
            , linearRegs(linearRegsVal)
            , halfLds(halfLdsVal)
            , r2c(r2cVal)
            , c2r(c2rVal)
            , rcFull(rcFullVal)
            , rcSimple(rcSimpleVal)
            , realSpecial(realSpecialVal)
            , enableGrouping(true)
            , numB1(0)
            , numB2(0)
            , numB4(0)
            , nextPass(NULL)
            , fft_doPreCallback(false)
            , fft_doPostCallback(false)
        {
            assert(radix <= length);
            assert(length % radix == 0);

            numButterfly  = cnPerWI / radix;
            workGroupSize = length / cnPerWI;

            // Total number of butterflies (over all work-tems) must be divisible by LS
            assert(((numButterfly * workGroupSize) % algLS) == 0);

            // All butterflies in one work-item should always be part of no more than 1
            // FFT transform.
            // In other words, there should not be more than 1 FFT transform per
            // work-item.
            assert(cnPerWI <= length);

            // Calculate the different types of Butterflies needed
            if(linearRegs || r2c || c2r)
            {
                numB1 = numButterfly;
            }
            else
            {
                numB4 = numButterfly / 4;
                numB2 = (numButterfly % 4) / 2; // can be 0 or 1
                numB1 = (numButterfly % 2); // can be 0 or 1

                assert(numButterfly == (numB4 * 4 + numB2 * 2 + numB1));
            }

            // if only half LDS can be used, we need the passes to share registers
            // and hence they need to be linear registers
            if(halfLds)
                assert(linearRegs);
        }

        size_t GetNumB1() const
        {
            return numB1;
        }
        size_t GetNumB2() const
        {
            return numB2;
        }
        size_t GetNumB4() const
        {
            return numB4;
        }

        size_t GetPosition() const
        {
            return position;
        }
        size_t GetRadix() const
        {
            return radix;
        }

        void SetNextPass(Pass<PR>* np)
        {
            nextPass = np;
        }
        void SetGrouping(bool grp)
        {
            enableGrouping = grp;
        }

        void SetPrecallback(bool hasPrecallback) // not used even
        {
            fft_doPreCallback = hasPrecallback;
        }

        void SetPostcallback(bool hasPostcallback) // not used even
        {
            fft_doPostCallback = hasPostcallback;
        }

        void GeneratePass(bool         fwd,
                          std::string  name_suffix,
                          std::string& passStr,
                          bool         fft_3StepTwiddle,
                          bool         twiddleFront,
                          bool         inInterleaved,
                          bool         outInterleaved,
                          bool         inReal,
                          bool         outReal,
                          size_t       inStride,
                          size_t       outStride,
                          double       scale,
                          bool         gIn  = false,
                          bool         gOut = false) const
        {
            const std::string bufferInRe  = (inReal || inInterleaved) ? "bufIn" : "bufInRe";
            const std::string bufferInIm  = (inReal || inInterleaved) ? "bufIn" : "bufInIm";
            const std::string bufferOutRe = (outReal || outInterleaved) ? "bufOut" : "bufOutRe";
            const std::string bufferOutIm = (outReal || outInterleaved) ? "bufOut" : "bufOutIm";

            const std::string bufferInRe2  = (inReal || inInterleaved) ? "bufIn2" : "bufInRe2";
            const std::string bufferInIm2  = (inReal || inInterleaved) ? "bufIn2" : "bufInIm2";
            const std::string bufferOutRe2 = (outReal || outInterleaved) ? "bufOut2" : "bufOutRe2";
            const std::string bufferOutIm2 = (outReal || outInterleaved) ? "bufOut2" : "bufOutIm2";

            // for real transforms we use only B1 butteflies (regC = 1)
            if(r2c || c2r)
            {
                assert(numB1 == numButterfly);
                assert(linearRegs);
            }

            // Check if it is single pass transform
            bool singlePass = ((position == 0) && (nextPass == NULL));
            if(singlePass)
                assert(numButterfly == 1); // for single pass transforms, there can be
            // only 1 butterfly per transform
            if(singlePass)
                assert(workGroupSize == 1);

            // Register types
            std::string regB1Type = RegBaseType<PR>(1);
            std::string regB2Type = RegBaseType<PR>(2);
            std::string regB4Type = RegBaseType<PR>(4);

            // Function attribute
            passStr += "template <typename T, StrideBin sb> \n";
            passStr += "__device__ void\n";

            // Function name
            passStr += PassName(position, fwd, length, name_suffix);

            // Function arguments
            passStr += "(const " + regB2Type + " *twiddles, ";
            if(name_suffix == "_sbcc")
                passStr += "const " + regB2Type
                           + " *twiddles_large, "; // the blockCompute BCT_C2C algorithm use
            // one more twiddle parameter
            passStr += "const size_t stride_in, const size_t stride_out, ";
            passStr += "unsigned int rw, unsigned int b, ";
            if(realSpecial)
                passStr += "unsigned int t, ";
            passStr += "unsigned int me, unsigned int inOffset, unsigned int outOffset, ";

            if(r2c || c2r)
            {
                assert(halfLds);

                if(gIn)
                {
                    if(inInterleaved)
                    {
                        passStr += regB2Type;
                        passStr += " *";
                        passStr += bufferInRe;
                        passStr += ", ";
                        if(!rcSimple)
                        {
                            passStr += regB2Type;
                            passStr += " *";
                            passStr += bufferInRe2;
                            passStr += ", ";
                        }
                    }
                    else if(inReal)
                    {
                        passStr += regB1Type;
                        passStr += " *";
                        passStr += bufferInRe;
                        passStr += ", ";
                        if(!rcSimple)
                        {
                            passStr += regB1Type;
                            passStr += " *";
                            passStr += bufferInRe2;
                            passStr += ", ";
                        }
                    }
                    else
                    {
                        passStr += regB1Type;
                        passStr += " *";
                        passStr += bufferInRe;
                        passStr += ", ";
                        if(!rcSimple)
                        {
                            passStr += regB1Type;
                            passStr += " *";
                            passStr += bufferInRe2;
                            passStr += ", ";
                        }
                        passStr += regB1Type;
                        passStr += " *";
                        passStr += bufferInIm;
                        passStr += ", ";
                        if(!rcSimple)
                        {
                            passStr += regB1Type;
                            passStr += " *";
                            passStr += bufferInIm2;
                            passStr += ", ";
                        }
                    }
                }
                else
                {
                    passStr += regB1Type;
                    passStr += " *";
                    passStr += bufferInRe;
                    passStr += ", ";
                    passStr += regB1Type;
                    passStr += " *";
                    passStr += bufferInIm;
                    passStr += ", ";
                }

                if(gOut)
                {
                    if(outInterleaved)
                    {
                        passStr += regB2Type;
                        passStr += " *";
                        passStr += bufferOutRe;
                        if(!rcSimple)
                        {
                            passStr += ", ";
                            passStr += regB2Type;
                            passStr += " *";
                            passStr += bufferOutRe2;
                        }
                    }
                    else if(outReal)
                    {
                        passStr += regB1Type;
                        passStr += " *";
                        passStr += bufferOutRe;
                        if(!rcSimple)
                        {
                            passStr += ", ";
                            passStr += regB1Type;
                            passStr += " *";
                            passStr += bufferOutRe2;
                        }
                    }
                    else
                    {
                        passStr += regB1Type;
                        passStr += " *";
                        passStr += bufferOutRe;
                        passStr += ", ";
                        if(!rcSimple)
                        {
                            passStr += regB1Type;
                            passStr += " *";
                            passStr += bufferOutRe2;
                            passStr += ", ";
                        }
                        passStr += regB1Type;
                        passStr += " *";
                        passStr += bufferOutIm;
                        if(!rcSimple)
                        {
                            passStr += ", ";
                            passStr += regB1Type;
                            passStr += " *";
                            passStr += bufferOutIm2;
                        }
                    }
                }
                else
                {
                    passStr += regB1Type;
                    passStr += " *";
                    passStr += bufferOutRe;
                    passStr += ", ";
                    passStr += regB1Type;
                    passStr += " *";
                    passStr += bufferOutIm;
                }
            }
            else
            {
                if(gIn)
                {
                    if(inInterleaved)
                    {
                        passStr += regB2Type;
                        passStr += " *";
                        passStr += bufferInRe;
                        passStr += ", ";
                    }
                    else
                    {
                        passStr += regB1Type;
                        passStr += " *";
                        passStr += bufferInRe;
                        passStr += ", ";
                        passStr += regB1Type;
                        passStr += " *";
                        passStr += bufferInIm;
                        passStr += ", ";
                    }
                }
                else
                {
                    if(inInterleaved)
                    {
                        passStr += regB2Type;
                        passStr += " *";
                        passStr += bufferInRe;
                        passStr += ", ";
                    }
                    else
                    {
                        passStr += regB1Type;
                        passStr += " *";
                        passStr += bufferInRe;
                        passStr += ", ";
                        passStr += regB1Type;
                        passStr += " *";
                        passStr += bufferInIm;
                        passStr += ", ";
                    }
                }

                if(gOut)
                {
                    if(outInterleaved)
                    {
                        passStr += regB2Type;
                        passStr += " *";
                        passStr += bufferOutRe;
                    }
                    else
                    {
                        passStr += regB1Type;
                        passStr += " *";
                        passStr += bufferOutRe;
                        passStr += ", ";
                        passStr += regB1Type;
                        passStr += " *";
                        passStr += bufferOutIm;
                    }
                }
                else
                {
                    if(outInterleaved)
                    {
                        passStr += regB2Type;
                        passStr += " *";
                        passStr += bufferOutRe;
                    }
                    else
                    {
                        passStr += regB1Type;
                        passStr += " *";
                        passStr += bufferOutRe;
                        passStr += ", ";
                        passStr += regB1Type;
                        passStr += " *";
                        passStr += bufferOutIm;
                    }
                }
            }

            // Register arguments
            if(linearRegs)
            {
                passStr += ", ";
                passStr += IterRegArgs();
            }

            passStr += ")\n{\n";

            // Register Declarations
            if(!linearRegs)
            {
                DeclareRegs(regB1Type, 1, numB1, passStr);
                DeclareRegs(regB2Type, 2, numB2, passStr);
                DeclareRegs(regB4Type, 4, numB4, passStr);
            }

            // odd cnPerWI processing
            bool oddp = false;
            oddp      = ((cnPerWI % 2) && (length > 1) && (!singlePass));

            // additional register for odd
            if(!rcSimple && oddp && ((r2c && (nextPass == NULL)) || (c2r && (position == 0))))
            {
                passStr += "\n\t";
                passStr += "unsigned int brv = 0;\n\t";
                passStr += "\n\t";
                passStr += regB2Type;
                passStr += " R";
                passStr += std::to_string(cnPerWI);
                passStr += "[1];\n\t";
                passStr += "(*R";
                passStr += std::to_string(cnPerWI);
                passStr += ").x = 0; ";
                passStr += "(*R";
                passStr += std::to_string(cnPerWI);
                passStr += ").y = 0;\n";
            }

            // Special private memory for c-r 1 pass transforms
            if(!rcSimple && (c2r && (position == 0)) && singlePass)
            {
                assert(radix == length);

                passStr += "\n\t";
                passStr += regB1Type;
                passStr += " mpvt[";
                passStr += std::to_string(length);
                passStr += "];\n";
            }

            passStr += "\n";

            // Read into registers
            if(r2c)
            {
                if(position == 0)
                {
                    passStr += "\n\tif(rw)\n\t{";
                    SweepRegs(SR_READ,
                              fwd,
                              inInterleaved,
                              inStride,
                              SR_COMP_REAL,
                              1.0f,
                              false,
                              bufferInRe,
                              bufferInIm,
                              "inOffset",
                              1,
                              numB1,
                              0,
                              passStr);
                    passStr += "\n\t}\n";

                    if(rcSimple)
                    {
                        passStr += "\n";
                        SweepRegsRC(SR_READ,
                                    fwd,
                                    inInterleaved,
                                    inStride,
                                    SR_COMP_IMAG,
                                    1.0f,
                                    true,
                                    true,
                                    false,
                                    bufferInRe2,
                                    bufferInIm2,
                                    "inOffset",
                                    passStr);
                        passStr += "\n";
                    }
                    else
                    {
                        passStr += "\n\tif(rw > 1)\n\t{";

                        SweepRegs(SR_READ,
                                  fwd,
                                  inInterleaved,
                                  inStride,
                                  SR_COMP_IMAG,
                                  1.0f,
                                  false,
                                  bufferInRe2,
                                  bufferInIm2,
                                  "inOffset",
                                  1,
                                  numB1,
                                  0,
                                  passStr);

                        passStr += "\n\t}\n";

                        passStr += "\telse\n\t{";
                        SweepRegsRC(SR_READ,
                                    fwd,
                                    inInterleaved,
                                    inStride,
                                    SR_COMP_IMAG,
                                    1.0f,
                                    true,
                                    true,
                                    false,
                                    bufferInRe2,
                                    bufferInIm2,
                                    "inOffset",
                                    passStr);
                        passStr += "\n\t}\n";
                    }
                }
            }
            else if(c2r && !rcSimple)
            {
                if(position == 0)
                {
                    std::string processBufRe     = bufferOutRe;
                    std::string processBufIm     = bufferOutIm;
                    std::string processBufOffset = "outOffset";
                    size_t      processBufStride = outStride;

                    if(singlePass)
                    {
                        processBufRe     = "mpvt";
                        processBufIm     = "mpvt";
                        processBufOffset = "0";
                        processBufStride = 1;
                    }

                    passStr += "\n\tif(rw && !me)\n\t{\n\t";
                    passStr += processBufRe;
                    passStr += "[";
                    passStr += processBufOffset;
                    passStr += "] = ";

                    passStr += bufferInRe;
                    passStr += "[inOffset]";

                    if(inInterleaved || fft_doPreCallback)
                        passStr += ".x;\n\t}";
                    else
                        passStr += ";\n\t}";

                    if(length > 1)
                    {
                        passStr += "\n\n\tif(rw)\n\t{";

                        SweepRegsRC(SR_READ,
                                    fwd,
                                    inInterleaved,
                                    inStride,
                                    SR_COMP_REAL,
                                    1.0f,
                                    false,
                                    false,
                                    false,
                                    bufferInRe,
                                    bufferInRe,
                                    "inOffset",
                                    passStr);

                        passStr += "\n\t}\n";

                        passStr += "\n\tif(rw > 1)\n\t{";

                        SweepRegsRC(SR_READ,
                                    fwd,
                                    inInterleaved,
                                    inStride,
                                    SR_COMP_REAL,
                                    1.0f,
                                    false,
                                    true,
                                    false,
                                    bufferInIm2,
                                    bufferInIm2,
                                    "inOffset",
                                    passStr);

                        passStr += "\n\t}\n\telse\n\t{";
                        SweepRegsRC(SR_READ,
                                    fwd,
                                    inInterleaved,
                                    inStride,
                                    SR_COMP_REAL,
                                    1.0f,
                                    true,
                                    true,
                                    false,
                                    bufferInIm2,
                                    bufferInIm2,
                                    "inOffset",
                                    passStr);
                        passStr += "\n\t}\n";

                        if(oddp)
                        {
                            passStr += "\n\tif(rw && (me%2))\n\t{";

                            SweepRegsRC(SR_READ,
                                        fwd,
                                        inInterleaved,
                                        inStride,
                                        SR_COMP_REAL,
                                        1.0f,
                                        false,
                                        false,
                                        true,
                                        bufferInRe,
                                        bufferInRe,
                                        "inOffset",
                                        passStr);

                            passStr += "\n\t}";
                            passStr += "\n\tif((rw > 1) && (me%2))\n\t{";

                            SweepRegsRC(SR_READ,
                                        fwd,
                                        inInterleaved,
                                        inStride,
                                        SR_COMP_REAL,
                                        1.0f,
                                        false,
                                        true,
                                        true,
                                        bufferInIm2,
                                        bufferInIm2,
                                        "inOffset",
                                        passStr);

                            passStr += "\n\t}\n";
                        }

                        SweepRegsRC(SR_WRITE,
                                    fwd,
                                    outInterleaved,
                                    processBufStride,
                                    SR_COMP_REAL,
                                    1.0f,
                                    false,
                                    true,
                                    false,
                                    processBufRe,
                                    processBufIm,
                                    processBufOffset,
                                    passStr);
                        if(oddp)
                        {
                            passStr += "\n\tif(me%2)\n\t{";
                            SweepRegsRC(SR_WRITE,
                                        fwd,
                                        outInterleaved,
                                        processBufStride,
                                        SR_COMP_REAL,
                                        1.0f,
                                        false,
                                        true,
                                        true,
                                        processBufRe,
                                        processBufIm,
                                        processBufOffset,
                                        passStr);
                            passStr += "\n\t}\n";
                        }
                        SweepRegsRC(SR_WRITE,
                                    fwd,
                                    outInterleaved,
                                    processBufStride,
                                    SR_COMP_REAL,
                                    1.0f,
                                    false,
                                    false,
                                    false,
                                    processBufRe,
                                    processBufIm,
                                    processBufOffset,
                                    passStr);
                        if(oddp)
                        {
                            passStr += "\n\tif(me%2)\n\t{";
                            SweepRegsRC(SR_WRITE,
                                        fwd,
                                        outInterleaved,
                                        processBufStride,
                                        SR_COMP_REAL,
                                        1.0f,
                                        false,
                                        false,
                                        true,
                                        processBufRe,
                                        processBufIm,
                                        processBufOffset,
                                        passStr);
                            passStr += "\n\t}\n";
                        }
                    }

                    passStr += "\n\n\t__syncthreads();\n";
                    SweepRegs(SR_READ,
                              fwd,
                              outInterleaved,
                              processBufStride,
                              SR_COMP_REAL,
                              1.0f,
                              false,
                              processBufRe,
                              processBufIm,
                              processBufOffset,
                              1,
                              numB1,
                              0,
                              passStr,
                              false,
                              oddp);
                    passStr += "\n\n\t__syncthreads();\n";

                    passStr += "\n\tif((rw > 1) && !me)\n\t{\n\t";
                    passStr += processBufIm;
                    passStr += "[";
                    passStr += processBufOffset;
                    passStr += "] = ";

                    passStr += bufferInRe2;
                    passStr += "[inOffset]";

                    if(inInterleaved || fft_doPreCallback)
                        passStr += ".x;\n\t}";
                    else
                        passStr += ";\n\t}";
                    passStr += "\n\tif((rw == 1) && !me)\n\t{\n\t";
                    passStr += processBufIm;
                    passStr += "[";
                    passStr += processBufOffset;
                    passStr += "] = 0;\n\t}";

                    if(length > 1)
                    {
                        if(!fft_doPreCallback)
                        {
                            passStr += "\n\n\tif(rw)\n\t{";
                            SweepRegsRC(SR_READ,
                                        fwd,
                                        inInterleaved,
                                        inStride,
                                        SR_COMP_IMAG,
                                        1.0f,
                                        false,
                                        false,
                                        false,
                                        bufferInIm,
                                        bufferInIm,
                                        "inOffset",
                                        passStr);
                            passStr += "\n\t}\n";

                            passStr += "\n\tif(rw > 1)\n\t{";
                            SweepRegsRC(SR_READ,
                                        fwd,
                                        inInterleaved,
                                        inStride,
                                        SR_COMP_IMAG,
                                        1.0f,
                                        false,
                                        true,
                                        false,
                                        bufferInRe2,
                                        bufferInRe2,
                                        "inOffset",
                                        passStr);
                            passStr += "\n\t}\n\telse\n\t{";
                            SweepRegsRC(SR_READ,
                                        fwd,
                                        inInterleaved,
                                        inStride,
                                        SR_COMP_IMAG,
                                        1.0f,
                                        true,
                                        true,
                                        false,
                                        bufferInRe2,
                                        bufferInRe2,
                                        "inOffset",
                                        passStr);
                            passStr += "\n\t}";

                            if(oddp)
                            {
                                passStr += "\n\tif(rw && (me%2))\n\t{";
                                SweepRegsRC(SR_READ,
                                            fwd,
                                            inInterleaved,
                                            inStride,
                                            SR_COMP_IMAG,
                                            1.0f,
                                            false,
                                            false,
                                            true,
                                            bufferInIm,
                                            bufferInIm,
                                            "inOffset",
                                            passStr);
                                passStr += "\n\t}";
                                passStr += "\n\tif((rw > 1) && (me%2))\n\t{";
                                SweepRegsRC(SR_READ,
                                            fwd,
                                            inInterleaved,
                                            inStride,
                                            SR_COMP_IMAG,
                                            1.0f,
                                            false,
                                            true,
                                            true,
                                            bufferInRe2,
                                            bufferInRe2,
                                            "inOffset",
                                            passStr);
                                passStr += "\n\t}";
                            }
                        }
                        passStr += "\n";

                        SweepRegsRC(SR_WRITE,
                                    fwd,
                                    outInterleaved,
                                    processBufStride,
                                    SR_COMP_IMAG,
                                    1.0f,
                                    false,
                                    true,
                                    false,
                                    processBufRe,
                                    processBufIm,
                                    processBufOffset,
                                    passStr);
                        if(oddp)
                        {
                            passStr += "\n\tif(me%2)\n\t{";
                            SweepRegsRC(SR_WRITE,
                                        fwd,
                                        outInterleaved,
                                        processBufStride,
                                        SR_COMP_IMAG,
                                        1.0f,
                                        false,
                                        true,
                                        true,
                                        processBufRe,
                                        processBufIm,
                                        processBufOffset,
                                        passStr);
                            passStr += "\n\t}\n";
                        }
                        SweepRegsRC(SR_WRITE,
                                    fwd,
                                    outInterleaved,
                                    processBufStride,
                                    SR_COMP_IMAG,
                                    1.0f,
                                    false,
                                    false,
                                    false,
                                    processBufRe,
                                    processBufIm,
                                    processBufOffset,
                                    passStr);
                        if(oddp)
                        {
                            passStr += "\n\tif(me%2)\n\t{";
                            SweepRegsRC(SR_WRITE,
                                        fwd,
                                        outInterleaved,
                                        processBufStride,
                                        SR_COMP_IMAG,
                                        1.0f,
                                        false,
                                        false,
                                        true,
                                        processBufRe,
                                        processBufIm,
                                        processBufOffset,
                                        passStr);
                            passStr += "\n\t}\n";
                        }
                    }

                    passStr += "\n\n\t__syncthreads();\n";
                    SweepRegs(SR_READ,
                              fwd,
                              outInterleaved,
                              processBufStride,
                              SR_COMP_IMAG,
                              1.0f,
                              false,
                              processBufRe,
                              processBufIm,
                              processBufOffset,
                              1,
                              numB1,
                              0,
                              passStr);
                    passStr += "\n\n\t__syncthreads();\n";
                }
            }
            else
            {
                if((!halfLds) || (halfLds && (position == 0)))
                {
                    bool isPrecallVector = false;

                    passStr += "\n\tif(rw)\n\t{";
                    SweepRegs(SR_READ,
                              fwd,
                              inInterleaved,
                              inStride,
                              SR_COMP_BOTH,
                              1.0f,
                              false,
                              bufferInRe,
                              bufferInIm,
                              "inOffset",
                              1,
                              numB1,
                              0,
                              passStr,
                              isPrecallVector);
                    SweepRegs(SR_READ,
                              fwd,
                              inInterleaved,
                              inStride,
                              SR_COMP_BOTH,
                              1.0f,
                              false,
                              bufferInRe,
                              bufferInIm,
                              "inOffset",
                              2,
                              numB2,
                              numB1,
                              passStr,
                              isPrecallVector);
                    SweepRegs(SR_READ,
                              fwd,
                              inInterleaved,
                              inStride,
                              SR_COMP_BOTH,
                              1.0f,
                              false,
                              bufferInRe,
                              bufferInIm,
                              "inOffset",
                              4,
                              numB4,
                              2 * numB2 + numB1,
                              passStr,
                              isPrecallVector);
                    passStr += "\n\t}\n";
                }
            }

            passStr += "\n";

            // 3-step twiddle multiplies done in the front
            bool tw3Done = false;
            if(fft_3StepTwiddle && twiddleFront)
            {
                tw3Done = true;
                if(linearRegs)
                {
                    SweepRegs(SR_TWMUL_3STEP,
                              fwd,
                              false,
                              1,
                              SR_COMP_BOTH,
                              1.0f,
                              true,
                              bufferInRe,
                              bufferInIm,
                              "",
                              1,
                              numB1,
                              0,
                              passStr);
                }
                else
                {
                    SweepRegs(SR_TWMUL_3STEP,
                              fwd,
                              false,
                              1,
                              SR_COMP_BOTH,
                              1.0f,
                              true,
                              bufferInRe,
                              bufferInIm,
                              "",
                              1,
                              numB1,
                              0,
                              passStr);
                    SweepRegs(SR_TWMUL_3STEP,
                              fwd,
                              false,
                              1,
                              SR_COMP_BOTH,
                              1.0f,
                              true,
                              bufferInRe,
                              bufferInIm,
                              "",
                              2,
                              numB2,
                              numB1,
                              passStr);
                    SweepRegs(SR_TWMUL_3STEP,
                              fwd,
                              false,
                              1,
                              SR_COMP_BOTH,
                              1.0f,
                              true,
                              bufferInRe,
                              bufferInIm,
                              "",
                              4,
                              numB4,
                              2 * numB2 + numB1,
                              passStr);
                }
            }

            passStr += "\n";

            // Twiddle multiply
            if((position > 0) && (radix > 1))
            {
                SweepRegs(SR_TWMUL,
                          fwd,
                          false,
                          1,
                          SR_COMP_BOTH,
                          1.0f,
                          false,
                          bufferInRe,
                          bufferInIm,
                          "",
                          1,
                          numB1,
                          0,
                          passStr);
                SweepRegs(SR_TWMUL,
                          fwd,
                          false,
                          1,
                          SR_COMP_BOTH,
                          1.0f,
                          false,
                          bufferInRe,
                          bufferInIm,
                          "",
                          2,
                          numB2,
                          numB1,
                          passStr);
                SweepRegs(SR_TWMUL,
                          fwd,
                          false,
                          1,
                          SR_COMP_BOTH,
                          1.0f,
                          false,
                          bufferInRe,
                          bufferInIm,
                          "",
                          4,
                          numB4,
                          2 * numB2 + numB1,
                          passStr);
            }

            // Butterfly calls
            if(radix > 1)
            {
                if(numB1)
                    CallButterfly(ButterflyName(radix, 1, fwd), 1, numB1, passStr);
                if(numB2)
                    CallButterfly(ButterflyName(radix, 2, fwd), 2, numB2, passStr);
                if(numB4)
                    CallButterfly(ButterflyName(radix, 4, fwd), 4, numB4, passStr);
            }

            if(!halfLds)
                passStr += "\n\n\t__syncthreads();\n";
            passStr += "\n\n";

            // 3-step twiddle multiplies
            if(fft_3StepTwiddle && !tw3Done)
            {
                assert(nextPass == NULL);
                if(linearRegs)
                {
                    SweepRegs(SR_TWMUL_3STEP,
                              fwd,
                              false,
                              1,
                              SR_COMP_BOTH,
                              1.0f,
                              false,
                              bufferInRe,
                              bufferInIm,
                              "",
                              1,
                              numB1,
                              0,
                              passStr);
                }
                else
                {
                    SweepRegs(SR_TWMUL_3STEP,
                              fwd,
                              false,
                              1,
                              SR_COMP_BOTH,
                              1.0f,
                              false,
                              bufferInRe,
                              bufferInIm,
                              "",
                              1,
                              numB1,
                              0,
                              passStr);
                    SweepRegs(SR_TWMUL_3STEP,
                              fwd,
                              false,
                              1,
                              SR_COMP_BOTH,
                              1.0f,
                              false,
                              bufferInRe,
                              bufferInIm,
                              "",
                              2,
                              numB2,
                              numB1,
                              passStr);
                    SweepRegs(SR_TWMUL_3STEP,
                              fwd,
                              false,
                              1,
                              SR_COMP_BOTH,
                              1.0f,
                              false,
                              bufferInRe,
                              bufferInIm,
                              "",
                              4,
                              numB4,
                              2 * numB2 + numB1,
                              passStr);
                }
            }

            // Write back from registers
            if(halfLds)
            {
                // In this case, we have to write & again read back for the next pass
                // since we are
                // using only half the lds. Number of barriers will increase at the cost
                // of halving the lds.

                if(nextPass == NULL) // last pass
                {
                    if(r2c && !rcSimple)
                    {
                        if(!singlePass)
                        {
                            SweepRegs(SR_WRITE,
                                      fwd,
                                      inInterleaved,
                                      inStride,
                                      SR_COMP_REAL,
                                      1.0f,
                                      false,
                                      bufferInRe,
                                      bufferInIm,
                                      "inOffset",
                                      1,
                                      numB1,
                                      0,
                                      passStr);
                            passStr += "\n\n\t__syncthreads();\n";
                            SweepRegsRC(SR_READ,
                                        fwd,
                                        inInterleaved,
                                        inStride,
                                        SR_COMP_REAL,
                                        1.0f,
                                        false,
                                        false,
                                        false,
                                        bufferInRe,
                                        bufferInIm,
                                        "inOffset",
                                        passStr);
                            if(oddp)
                            {
                                passStr += "\n\tif(me%2)\n\t{";
                                SweepRegsRC(SR_READ,
                                            fwd,
                                            inInterleaved,
                                            inStride,
                                            SR_COMP_REAL,
                                            1.0f,
                                            false,
                                            false,
                                            true,
                                            bufferInRe,
                                            bufferInIm,
                                            "inOffset",
                                            passStr);
                                passStr += "\n\t}\n";
                            }

                            passStr += "\n\tif(rw && !me)\n\t{\n\t";
                            if(outInterleaved)
                            {

                                passStr += bufferOutRe;
                                passStr += "[outOffset].x = ";
                                passStr += bufferInRe;
                                passStr += "[inOffset]";
                                if(scale != 1.0)
                                {
                                    passStr += " * ";
                                    passStr += FloatToStr(scale);
                                    passStr += FloatSuffix<PR>();
                                }
                                passStr += ";\n\t";
                                passStr += bufferOutIm;
                                passStr += "[outOffset].y = ";
                                passStr += "0;\n\t}";
                            }
                            else
                            {

                                passStr += bufferOutRe;
                                passStr += "[outOffset] = ";
                                passStr += bufferInRe;
                                passStr += "[inOffset]";
                                if(scale != 1.0)
                                {
                                    passStr += " * ";
                                    passStr += FloatToStr(scale);
                                    passStr += FloatSuffix<PR>();
                                }
                                passStr += ";\n\t";
                                passStr += bufferOutIm;
                                passStr += "[outOffset] = ";
                                passStr += "0;\n\t}";
                            }
                            passStr += "\n\n\t__syncthreads();\n";

                            SweepRegs(SR_WRITE,
                                      fwd,
                                      inInterleaved,
                                      inStride,
                                      SR_COMP_IMAG,
                                      1.0f,
                                      false,
                                      bufferInRe,
                                      bufferInIm,
                                      "inOffset",
                                      1,
                                      numB1,
                                      0,
                                      passStr);
                            passStr += "\n\n\t__syncthreads();\n";
                            SweepRegsRC(SR_READ,
                                        fwd,
                                        inInterleaved,
                                        inStride,
                                        SR_COMP_IMAG,
                                        1.0f,
                                        false,
                                        false,
                                        false,
                                        bufferInRe,
                                        bufferInIm,
                                        "inOffset",
                                        passStr);
                            if(oddp)
                            {
                                passStr += "\n\tif(me%2)\n\t{";
                                SweepRegsRC(SR_READ,
                                            fwd,
                                            inInterleaved,
                                            inStride,
                                            SR_COMP_IMAG,
                                            1.0f,
                                            false,
                                            false,
                                            true,
                                            bufferInRe,
                                            bufferInIm,
                                            "inOffset",
                                            passStr);
                                passStr += "\n\t}\n";
                            }

                            passStr += "\n\tif((rw > 1) && !me)\n\t{\n\t";
                            if(outInterleaved)
                            {

                                passStr += bufferOutRe2;
                                passStr += "[outOffset].x = ";
                                passStr += bufferInIm;
                                passStr += "[inOffset]";
                                if(scale != 1.0)
                                {
                                    passStr += " * ";
                                    passStr += FloatToStr(scale);
                                    passStr += FloatSuffix<PR>();
                                }
                                passStr += ";\n\t";
                                passStr += bufferOutIm2;
                                passStr += "[outOffset].y = ";
                                passStr += "0;\n\t}";
                            }
                            else
                            {

                                passStr += bufferOutRe2;
                                passStr += "[outOffset] = ";
                                passStr += bufferInIm;
                                passStr += "[inOffset]";
                                if(scale != 1.0)
                                {
                                    passStr += " * ";
                                    passStr += FloatToStr(scale);
                                    passStr += FloatSuffix<PR>();
                                }
                                passStr += ";\n\t";
                                passStr += bufferOutIm2;
                                passStr += "[outOffset] = ";
                                passStr += "0;\n\t}";
                            }
                            passStr += "\n\n\t__syncthreads();\n";
                        }

                        passStr += "\n\n\tif(rw)\n\t{";
                        SweepRegsRC(SR_WRITE,
                                    fwd,
                                    outInterleaved,
                                    outStride,
                                    SR_COMP_BOTH,
                                    scale,
                                    false,
                                    false,
                                    false,
                                    bufferOutRe,
                                    bufferOutIm,
                                    "outOffset",
                                    passStr);
                        passStr += "\n\t}\n";
                        if(oddp)
                        {
                            passStr += "\n\n\tbrv = ((rw != 0) & (me%2 == 1));\n\t";
                            passStr += "if(brv)\n\t{";
                            SweepRegsRC(SR_WRITE,
                                        fwd,
                                        outInterleaved,
                                        outStride,
                                        SR_COMP_BOTH,
                                        scale,
                                        false,
                                        false,
                                        true,
                                        bufferOutRe,
                                        bufferOutIm,
                                        "outOffset",
                                        passStr);
                            passStr += "\n\t}\n";
                        }

                        passStr += "\n\n\tif(rw > 1)\n\t{";

                        std::string outOffset;
                        outOffset += "outOffset";

                        SweepRegsRC(SR_WRITE,
                                    fwd,
                                    outInterleaved,
                                    outStride,
                                    SR_COMP_BOTH,
                                    scale,
                                    false,
                                    true,
                                    false,
                                    bufferOutRe2,
                                    bufferOutIm2,
                                    outOffset,
                                    passStr);
                        passStr += "\n\t}\n";
                        if(oddp)
                        {
                            passStr += "\n\n\tbrv = ((rw > 1) & (me%2 == 1));\n\t";
                            passStr += "if(brv)\n\t{";
                            SweepRegsRC(SR_WRITE,
                                        fwd,
                                        outInterleaved,
                                        outStride,
                                        SR_COMP_BOTH,
                                        scale,
                                        false,
                                        true,
                                        true,
                                        bufferOutRe2,
                                        bufferOutIm2,
                                        outOffset,
                                        passStr);
                            passStr += "\n\t}\n";
                        }
                    }
                    else if(c2r)
                    {
                        passStr += "\n\tif(rw)\n\t{";
                        SweepRegs(SR_WRITE,
                                  fwd,
                                  outInterleaved,
                                  outStride,
                                  SR_COMP_REAL,
                                  scale,
                                  false,
                                  bufferOutRe,
                                  bufferOutIm,
                                  "outOffset",
                                  1,
                                  numB1,
                                  0,
                                  passStr);
                        passStr += "\n\t}\n";

                        if(!rcSimple)
                        {
                            std::string outOffset;
                            outOffset += "outOffset";

                            passStr += "\n\tif(rw > 1)\n\t{";
                            SweepRegs(SR_WRITE,
                                      fwd,
                                      outInterleaved,
                                      outStride,
                                      SR_COMP_IMAG,
                                      scale,
                                      false,
                                      bufferOutRe2,
                                      bufferOutIm2,
                                      outOffset,
                                      1,
                                      numB1,
                                      0,
                                      passStr);
                            passStr += "\n\t}\n";
                        }
                    }
                    else
                    {
                        passStr += "\n\tif(rw)\n\t{";
                        SweepRegs(SR_WRITE,
                                  fwd,
                                  outInterleaved,
                                  outStride,
                                  SR_COMP_BOTH,
                                  scale,
                                  false,
                                  bufferOutRe,
                                  bufferOutIm,
                                  "outOffset",
                                  1,
                                  numB1,
                                  0,
                                  passStr);
                        passStr += "\n\t}\n";
                    }
                }
                else
                {
                    passStr += "\n\tif(rw)\n\t{";
                    SweepRegs(SR_WRITE,
                              fwd,
                              outInterleaved,
                              outStride,
                              SR_COMP_REAL,
                              scale,
                              false,
                              bufferOutRe,
                              bufferOutIm,
                              "outOffset",
                              1,
                              numB1,
                              0,
                              passStr);
                    passStr += "\n\t}\n";
                    passStr += "\n\n\t__syncthreads();\n";
                    passStr += "\n\tif(rw)\n\t{";
                    nextPass->SweepRegs(SR_READ,
                                        fwd,
                                        outInterleaved,
                                        outStride,
                                        SR_COMP_REAL,
                                        scale,
                                        false,
                                        bufferOutRe,
                                        bufferOutIm,
                                        "outOffset",
                                        1,
                                        nextPass->GetNumB1(),
                                        0,
                                        passStr);
                    passStr += "\n\t}\n";
                    passStr += "\n\n\t__syncthreads();\n";
                    passStr += "\n\tif(rw)\n\t{";
                    SweepRegs(SR_WRITE,
                              fwd,
                              outInterleaved,
                              outStride,
                              SR_COMP_IMAG,
                              scale,
                              false,
                              bufferOutRe,
                              bufferOutIm,
                              "outOffset",
                              1,
                              numB1,
                              0,
                              passStr);
                    passStr += "\n\t}\n";
                    passStr += "\n\n\t__syncthreads();\n";
                    passStr += "\n\tif(rw)\n\t{";
                    nextPass->SweepRegs(SR_READ,
                                        fwd,
                                        outInterleaved,
                                        outStride,
                                        SR_COMP_IMAG,
                                        scale,
                                        false,
                                        bufferOutRe,
                                        bufferOutIm,
                                        "outOffset",
                                        1,
                                        nextPass->GetNumB1(),
                                        0,
                                        passStr);
                    passStr += "\n\t}\n";
                    passStr += "\n\n\t__syncthreads();\n";
                }
            }
            else
            {

                passStr += "\n\tif(rw)\n\t{";
                SweepRegs(SR_WRITE,
                          fwd,
                          outInterleaved,
                          outStride,
                          SR_COMP_BOTH,
                          scale,
                          false,
                          bufferOutRe,
                          bufferOutIm,
                          "outOffset",
                          1,
                          numB1,
                          0,
                          passStr);
                SweepRegs(SR_WRITE,
                          fwd,
                          outInterleaved,
                          outStride,
                          SR_COMP_BOTH,
                          scale,
                          false,
                          bufferOutRe,
                          bufferOutIm,
                          "outOffset",
                          2,
                          numB2,
                          numB1,
                          passStr);
                SweepRegs(SR_WRITE,
                          fwd,
                          outInterleaved,
                          outStride,
                          SR_COMP_BOTH,
                          scale,
                          false,
                          bufferOutRe,
                          bufferOutIm,
                          "outOffset",
                          4,
                          numB4,
                          2 * numB2 + numB1,
                          passStr);
                passStr += "\n\t}\n";
            }

            passStr += "\n}\n\n";
        }
    };
};

#endif
