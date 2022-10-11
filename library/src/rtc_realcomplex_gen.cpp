// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rtc_realcomplex_gen.h"
#include "../../shared/array_predicate.h"
#include "device/generator/generator.h"

#include "device/kernel-generator-embed.h"

// generate name for RTC realcomplex kernel
std::string realcomplex_rtc_kernel_name(const RealComplexSpecs& specs)
{
    std::string kernel_name;

    switch(specs.scheme)
    {
    case CS_KERNEL_COPY_R_TO_CMPLX:
        kernel_name += "r2c_copy_rtc";
        break;
    case CS_KERNEL_COPY_CMPLX_TO_HERM:
        kernel_name += "c2herm_copy_rtc";
        break;
    case CS_KERNEL_COPY_CMPLX_TO_R:
        kernel_name += "c2r_copy_rtc";
        break;
    case CS_KERNEL_COPY_HERM_TO_CMPLX:
        kernel_name += "herm2c_copy_rtc";
        break;
    default:
        throw std::runtime_error("invalid realcomplex rtc scheme");
    }

    kernel_name += "_dim" + std::to_string(specs.dim);

    kernel_name += rtc_precision_name(specs.precision);
    kernel_name += rtc_array_type_name(specs.inArrayType);
    kernel_name += rtc_array_type_name(specs.outArrayType);

    if(specs.enable_callbacks)
        kernel_name += "_CB";
    if(specs.enable_scaling)
        kernel_name += "_scale";

    return kernel_name;
}

std::string r2c_copy_rtc(const std::string& kernel_name, const RealComplexSpecs& specs)
{
    std::string src;

    const char* input_type
        = specs.scheme == CS_KERNEL_COPY_R_TO_CMPLX ? "real_type_t<scalar_type>" : "scalar_type";
    const char* output_type
        = specs.scheme == CS_KERNEL_COPY_CMPLX_TO_R ? "real_type_t<scalar_type>" : "scalar_type";

    // function arguments
    Variable hermitian_size{"hermitian_size", "const unsigned int"};
    Variable lengths0{"lengths0", "unsigned int"};
    Variable lengths1{"lengths1", "unsigned int"};
    Variable lengths2{"lengths2", "unsigned int"};
    Variable stride_in0{"stride_in0", "unsigned int"};
    Variable stride_in1{"stride_in1", "unsigned int"};
    Variable stride_in2{"stride_in2", "unsigned int"};
    Variable stride_in3{"stride_in3", "unsigned int"};
    Variable stride_out0{"stride_out0", "unsigned int"};
    Variable stride_out1{"stride_out1", "unsigned int"};
    Variable stride_out2{"stride_out2", "unsigned int"};
    Variable stride_out3{"stride_out3", "unsigned int"};
    Variable input{"input", input_type, true, true};
    Variable output{"output", output_type, true, true};
    Variable scale_factor_var{"scale_factor", "const real_type_t<scalar_type>"};

    Function func(kernel_name);
    func.launch_bounds = LAUNCH_BOUNDS_R2C_C2R_KERNEL;
    func.qualifier     = "extern \"C\" __global__";

    if(specs.scheme == CS_KERNEL_COPY_HERM_TO_CMPLX)
        func.arguments.append(hermitian_size);

    func.arguments.append(lengths0);
    func.arguments.append(lengths1);
    func.arguments.append(lengths2);
    func.arguments.append(stride_in0);
    func.arguments.append(stride_in1);
    func.arguments.append(stride_in2);
    func.arguments.append(stride_in3);
    func.arguments.append(stride_out0);
    func.arguments.append(stride_out1);
    func.arguments.append(stride_out2);
    func.arguments.append(stride_out3);
    func.arguments.append(input);
    func.arguments.append(output);
    for(const auto& arg : get_callback_args().arguments)
        func.arguments.append(arg);
    if(specs.scheme == CS_KERNEL_COPY_CMPLX_TO_HERM || specs.scheme == CS_KERNEL_COPY_CMPLX_TO_R)
        func.arguments.append(scale_factor_var);

    Variable dim_var{"dim", "const unsigned int"};

    Variable idx_0{"idx_0", "const unsigned int"};
    func.body += Declaration{idx_0, "blockIdx.x * blockDim.x + threadIdx.x"};

    if(specs.scheme == CS_KERNEL_COPY_HERM_TO_CMPLX)
    {
        Variable input_offset{"input_offset", "auto"};
        Variable outputs_offset{"outputs_offset", "auto"};
        Variable outputc_offset{"outputc_offset", "auto"};
        func.body += CommentLines{"start with batch offset"};
        func.body
            += Declaration{input_offset, "blockIdx.z * stride_in" + std::to_string(specs.dim)};
        func.body += CommentLines{"straight copy"};
        func.body
            += Declaration{outputs_offset, "blockIdx.z * stride_out" + std::to_string(specs.dim)};
        func.body += CommentLines{"conjugate copy"};
        func.body
            += Declaration{outputc_offset, "blockIdx.z * stride_out" + std::to_string(specs.dim)};

        func.body += CommentLines{"straight copy indices"};
        Variable is0{"is0", "auto"};
        Variable is1{"is1", "auto"};
        Variable is2{"is2", "auto"};
        func.body += Declaration{is0, idx_0};
        func.body += Declaration{is1, Literal{"blockIdx.y"} % lengths1};
        func.body += Declaration{is2, Literal{"blockIdx.y"} / lengths1};

        func.body += CommentLines{"conjugate copy indices"};
        Variable ic0{"ic0", "auto"};
        Variable ic1{"ic1", "auto"};
        Variable ic2{"ic2", "auto"};
        func.body += Declaration{ic0, Ternary{is0 == 0, 0, lengths0 - is0}};
        func.body += Declaration{ic1, Ternary{is1 == 0, 0, lengths1 - is1}};
        func.body += Declaration{ic2, Ternary{is2 == 0, 0, lengths2 - is2}};

        func.body
            += AddAssign(input_offset, is2 * stride_in2 + is1 * stride_in1 + is0 * stride_in0);

        func.body += CommentLines{
            "notice for 1D, blockIdx.y == 0 and thus has no effect for input_offset"};
        func.body
            += AddAssign(outputs_offset, is2 * stride_out2 + is1 * stride_out1 + is0 * stride_out0);
        func.body
            += AddAssign(outputc_offset, ic2 * stride_out2 + ic1 * stride_out1 + ic0 * stride_out0);

        func.body += CallbackDeclaration("scalar_type", "cbtype");

        func.body += CommentLines{"we would do hermitian2complex at the start of a C2R transform,",
                                  "so it would never be the last kernel to write to global",
                                  "memory.  don't bother going through the store callback to",
                                  "write global memory."};
        Variable outputs{"outputs", "scalar_type", true};
        Variable outputc{"outputc", "scalar_type", true};
        func.body += Declaration{outputs, output + outputs_offset};
        func.body += Declaration{outputc, output + outputc_offset};

        func.body += CommentLines{"simply write the element to output"};
        If write_simple{Or{is0 == 0, is0 * 2 == lengths0}, {}};
        write_simple.body += CommentLines{"simply write the element to output"};
        write_simple.body += Assign{outputs[0], LoadGlobal{input, input_offset}};
        write_simple.body += Return{};
        func.body += write_simple;

        If write_conj{is0 < hermitian_size, {}};

        Variable elem{"elem", "scalar_type"};
        write_conj.body += Declaration{elem};
        write_conj.body += Assign{elem, LoadGlobal{input, input_offset}};
        write_conj.body += Assign{outputs[0], elem};
        write_conj.body += Assign{elem.y, UnaryMinus{elem.y}};
        write_conj.body += Assign{outputc[0], elem};
        func.body += write_conj;
    }
    else
    {
        Variable lengths{"lengths", "const unsigned int", false, false, 3};
        Variable stride_in{"stride_in", "const unsigned int", false, false, 4};
        Variable stride_out{"stride_out", "const unsigned int", false, false, 4};
        func.body += Declaration{lengths, ComplexLiteral{lengths0, lengths1, lengths2}};
        func.body += Declaration{stride_in,
                                 ComplexLiteral{stride_in0, stride_in1, stride_in2, stride_in3}};
        func.body += Declaration{
            stride_out, ComplexLiteral{stride_out0, stride_out1, stride_out2, stride_out3}};

        func.body += CommentLines{"offsets"};
        Variable offset_in{"offset_in", "size_t"};
        Variable offset_out{"offset_out", "size_t"};
        Variable remaining{"remaining", "size_t"};
        Variable index_along_d{"index_along_d", "size_t"};
        func.body += Declaration{offset_in, 0};
        func.body += Declaration{offset_out, 0};
        func.body += Declaration{remaining, "blockIdx.y"};
        func.body += Declaration{index_along_d};
        Variable d{"d", "unsigned int"};
        For      offset_loop{d, 1, d < dim_var, 1};
        offset_loop.body += Assign{index_along_d, remaining % lengths[d]};
        offset_loop.body += Assign{remaining, remaining / lengths[d]};
        offset_loop.body += Assign{offset_in, offset_in + index_along_d * stride_in[d]};
        offset_loop.body += Assign{offset_out, offset_out + index_along_d * stride_out[d]};
        func.body += offset_loop;

        func.body += CommentLines{
            "remaining should be 1 at this point, since batch goes into blockIdx.z"};
        Variable batch{"batch", "unsigned int"};
        func.body += Declaration{batch, "blockIdx.z"};
        func.body += Assign{offset_in, offset_in + batch * stride_in[dim_var]};
        func.body += Assign{offset_out, offset_out + batch * stride_out[dim_var]};

        Variable      inputIdx{"inputIdx", "auto"};
        Variable      outputIdx{"outputIdx", "auto"};
        StatementList indexes{Declaration{inputIdx, offset_in + idx_0 * stride_in[0]},
                              Declaration{outputIdx, offset_out + idx_0 * stride_out[0]}};

        if(specs.scheme == CS_KERNEL_COPY_R_TO_CMPLX)
        {
            If guard{idx_0 < lengths[0], indexes};
            guard.body += CommentLines{"we would do real2complex at the beginning of an R2C",
                                       "transform, so it would never be the last kernel to write",
                                       "to global memory.  don't bother going through the store cb",
                                       "to write global memory."};
            guard.body += CallbackDeclaration("real_type_t<scalar_type>", "cbtype");

            ComplexLiteral elem{LoadGlobal{input, inputIdx}, "0.0"};
            guard.body += Assign{output[outputIdx], elem};
            func.body += guard;
        }
        else if(specs.scheme == CS_KERNEL_COPY_CMPLX_TO_HERM)
        {
            func.body += CommentLines{"only read and write the first [length0/2+1] elements "
                                      "due to conjugate redundancy"};
            If guard{idx_0 < Parens{1 + lengths[0] / 2}, indexes};
            guard.body += CommentLines{"we would do complex2hermitian at the end of an R2C",
                                       "transform, so it would never be the first kernel to read",
                                       "from global memory.  don't bother going through the load",
                                       "callback to read global memory."};

            guard.body += CallbackDeclaration("scalar_type", "cbtype");

            Variable elem{"elem", "scalar_type"};
            guard.body += Declaration{elem, input[inputIdx]};
            if(specs.enable_scaling)
                guard.body += MultiplyAssign(elem, scale_factor_var);
            guard.body += StoreGlobal{output, outputIdx, elem};
            func.body += guard;
        }
        else if(specs.scheme == CS_KERNEL_COPY_CMPLX_TO_R)
        {
            If guard{idx_0 < lengths[0], indexes};
            guard.body
                += CommentLines{"we would do complex2real at the end of a C2R",
                                "transform, so it would never be the first kernel to read",
                                "from global memory.  don't bother going through the load cb",
                                "to read global memory."};
            guard.body += CallbackDeclaration("real_type_t<scalar_type>", "cbtype");

            Variable elem{"elem", "auto"};
            guard.body += Declaration{elem, input[inputIdx].x};
            if(specs.enable_scaling)
                guard.body += MultiplyAssign(elem, scale_factor_var);
            guard.body += StoreGlobal{output, outputIdx, elem};
            func.body += guard;
        }
    }

    if(array_type_is_planar(specs.inArrayType))
        func = make_planar(func, "input");
    if(array_type_is_planar(specs.outArrayType))
        func = make_planar(func, "output");

    src += func.render();
    return src;
}

// generate source for RTC realcomplex kernel.
std::string realcomplex_rtc(const std::string& kernel_name, const RealComplexSpecs& specs)
{
    std::string src;
    // includes and declarations
    src += common_h;
    src += callback_h;

    src += rtc_precision_type_decl(specs.precision);

    src += rtc_const_cbtype_decl(specs.enable_callbacks);

    src += "static const unsigned int dim = " + std::to_string(specs.dim) + ";\n";

    switch(specs.scheme)
    {
    case CS_KERNEL_COPY_R_TO_CMPLX:
    case CS_KERNEL_COPY_CMPLX_TO_HERM:
    case CS_KERNEL_COPY_CMPLX_TO_R:
    case CS_KERNEL_COPY_HERM_TO_CMPLX:
        return src + r2c_copy_rtc(kernel_name, specs);
    default:
        throw std::runtime_error("invalid realcomplex rtc scheme");
    }
}
