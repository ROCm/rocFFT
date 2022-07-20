// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <functional>
using namespace std::placeholders;

#include "generator.h"
#include "stockham_gen.h"
#include <array>
#include <fstream>
#include <iostream>
#include <optional>

#include "stockham_gen_cc.h"
#include "stockham_gen_cr.h"
#include "stockham_gen_rc.h"
#include "stockham_gen_rr.h"

#include "stockham_gen_2d.h"

std::string make_place_format_variants(const Function&                device,
                                       const std::optional<Function>& device1,
                                       const Function&                global,
                                       bool                           allow_inplace = true)
{
    std::string output;

    // device functions have no difference between ip/op
    output += device.render();
    if(device1)
        output += (*device1).render();

    // inplace, interleaved
    if(allow_inplace)
    {
        output += make_inplace(global).render();

        // in-place, planar
        output += make_inplace(make_planar(global, "buf")).render();
    }

    // out-of-place, interleaved -> interleaved
    auto global_outplace = make_outofplace(global);

    output += global_outplace.render();

    // out-of-place, interleaved -> planar
    auto global_outplace_planar_out = make_planar(global_outplace, "buf_out");
    output += global_outplace_planar_out.render();

    // out-of-place, planar -> interleaved
    output += make_planar(global_outplace, "buf_in").render();

    // out-of-place, planar -> planar
    output += make_planar(global_outplace_planar_out, "buf_in").render();
    return output;
}

// this rolls up all the information about the generated launchers,
// enough to genernate the function pool entry
struct GeneratedLauncher
{
    GeneratedLauncher(StockhamKernel&    kernel,
                      const std::string& scheme,
                      const std::string& name,
                      bool               double_precision,
                      const std::string& sbrc_type,
                      const std::string& sbrc_transpose_type)
        : name(name)
        , scheme(scheme)
        , lengths(kernel.launcher_lengths())
        , factors(kernel.launcher_factors())
        , transforms_per_block(kernel.transforms_per_block)
        , workgroup_size(kernel.workgroup_size)
        , half_lds(kernel.half_lds)
        , direct_to_from_reg(kernel.direct_to_from_reg)
        , sbrc_type(sbrc_type)
        , sbrc_transpose_type(sbrc_transpose_type)
        , double_precision(double_precision)
    {
    }

    std::string               name;
    std::string               scheme;
    std::vector<unsigned int> lengths;
    std::vector<unsigned int> factors;

    unsigned int transforms_per_block;
    unsigned int workgroup_size;
    bool         half_lds;
    bool         direct_to_from_reg;

    // SBRC transpose type
    std::string sbrc_type;
    std::string sbrc_transpose_type;
    bool        double_precision;

    // output a json object that the python generator can parse to know
    // how to build the function pool
    std::string to_string() const
    {
        std::string output = "{";

        const char* OBJ_DELIM = "";
        const char* COMMA     = ",";

        auto quote_str  = [](const std::string& s) { return "\"" + s + "\""; };
        auto add_member = [&](const std::string& key, const std::string& value) {
            output += OBJ_DELIM;
            output += quote_str(key) + ": " + value;
            OBJ_DELIM = COMMA;
        };
        auto vec_to_list = [&](const std::vector<unsigned int>& vec) {
            const char* LIST_DELIM = "";
            std::string list_str   = "[";
            for(auto i : vec)
            {
                list_str += LIST_DELIM;
                list_str += std::to_string(i);
                LIST_DELIM = COMMA;
            }
            list_str += "]";
            return list_str;
        };

        add_member("name", quote_str(name));
        add_member("scheme", quote_str(scheme));
        add_member("factors", vec_to_list(factors));
        add_member("lengths", vec_to_list(lengths));
        add_member("transforms_per_block", std::to_string(transforms_per_block));
        add_member("workgroup_size", std::to_string(workgroup_size));
        add_member("half_lds", half_lds ? "true" : "false");
        add_member("direct_to_from_reg", direct_to_from_reg ? "true" : "false");
        add_member("sbrc_type", quote_str(sbrc_type));
        add_member("sbrc_transpose_type", quote_str(sbrc_transpose_type));
        add_member("double_precision", double_precision ? "true" : "false");

        output += "}";
        return output;
    }
};

struct LaunchSuffix
{
    std::string function_suffix;
    std::string scheme;
    std::string sbrc_type;
    std::string sbrc_transpose_type;
};

// make launcher using POWX macro
std::string make_launcher(unsigned int                     length,
                          bool                             allow_inplace,
                          const std::vector<unsigned int>& precision_types,
                          const char*                      macro,
                          const std::vector<LaunchSuffix>& launcher_suffixes,
                          const std::string&               kernel_suffix,
                          StockhamKernel&                  kernel,
                          std::vector<GeneratedLauncher>&  generated_launchers)
{
    std::string       output;
    auto              length_str      = std::to_string(length);
    static const auto placements_both = {"ip", "op"};
    static const auto placements_op   = {"op"};
    static const auto directions      = {"forward", "inverse"};

    static const std::array<std::array<const char*, 2>, 2> precisions{
        {{"float", "sp"}, {"double", "dp"}}};

    for(auto precision_type : precision_types)
    {
        auto&& precision = precisions[precision_type];
        for(auto&& launcher : launcher_suffixes)
        {
            std::string launcher_name = "rocfft_internal_dfn_";
            launcher_name += precision[1];
            // SBRC specifically names the launchers with _op for some reason
            if(kernel_suffix == "SBRC")
                launcher_name += "_op";
            launcher_name += "_ci_ci_" + launcher.function_suffix + "_" + length_str;

            output += std::string(macro) + "(" + launcher_name;
            for(auto&& placement : allow_inplace ? placements_both : placements_op)
            {
                for(auto&& direction : directions)
                {
                    output += std::string(",") + placement + "_" + direction + "_length"
                              + length_str + "_" + kernel_suffix;
                }
            }
            output += std::string(",") + precision[0] + "2";
            if(!launcher.sbrc_type.empty())
                output += "," + launcher.sbrc_type;
            if(!launcher.sbrc_transpose_type.empty())
                output += "," + launcher.sbrc_transpose_type;
            output += ");\n";

            generated_launchers.emplace_back(kernel,
                                             launcher.scheme,
                                             launcher_name,
                                             precision_type == rocfft_precision_double,
                                             launcher.sbrc_type,
                                             launcher.sbrc_transpose_type);
        }
    }
    return output;
}

std::string append_headers()
{
    std::string output;

    // includes
    output += "#include \"kernel_launch.h\"\n";
    output += "#include \"kernels/butterfly_constant.h\"\n";
    output += "#include \"kernels/common.h\"\n";
    output += "#include \"real2complex_device.h\"\n";
    output += "#include \"rocfft_butterfly_template.h\"\n";
    output += "#include <hip/hip_runtime.h>\n\n";

    return output;
}

std::string append_common_functions(const Function&                device_load_lds,
                                    const Function&                device_store_lds,
                                    const std::optional<Function>& device_load_lds1,
                                    const std::optional<Function>& device_store_lds1)
{
    std::string output;

    output += device_load_lds.render();
    output += device_store_lds.render();

    if(device_load_lds1)
        output += (*device_load_lds1).render();

    if(device_store_lds1)
        output += (*device_store_lds1).render();

    return output;
}

std::string make_variants(const Function&                device,
                          const std::optional<Function>& device1,
                          const Function&                global,
                          bool                           allow_inplace)
{
    std::string output;

    // forward kernels
    output += make_place_format_variants(device, device1, global, allow_inplace);

    // inverse kernels
    output += make_place_format_variants(make_inverse(device),
                                         device1 ? make_inverse(*device1) : device1,
                                         make_inverse(global),
                                         allow_inplace);

    return output;
}

std::string stockham_variants(const std::string&      filename,
                              StockhamGeneratorSpecs& specs,
                              StockhamGeneratorSpecs& specs2d)
{
    std::vector<GeneratedLauncher> launchers;
    std::string                    output;
    output += append_headers();
    if(specs.scheme == "CS_KERNEL_STOCKHAM")
    {
        StockhamKernelRR kernel(specs);
        output += append_common_functions(kernel.generate_lds_to_reg_input_function(),
                                          kernel.generate_lds_from_reg_output_function(),
                                          {},
                                          {});
        output += make_variants(kernel.generate_device_function_with_bank_shift(),
                                {},
                                kernel.generate_global_function(),
                                true);
        output += make_launcher(specs.length,
                                true,
                                specs.precisions,
                                "POWX_SMALL_GENERATOR",
                                {{"stoc", specs.scheme, "", ""}},
                                "SBRR",
                                kernel,
                                launchers);
    }
    else if(specs.scheme == "CS_KERNEL_STOCKHAM_BLOCK_CC")
    {
        StockhamKernelCC kernel(specs);
        output += append_common_functions(kernel.generate_lds_to_reg_input_function(),
                                          kernel.generate_lds_from_reg_output_function(),
                                          {},
                                          {});
        output += make_variants(
            kernel.generate_device_function(), {}, kernel.generate_global_function(), true);
        output += make_launcher(specs.length,
                                true,
                                specs.precisions,
                                "POWX_LARGE_SBCC_GENERATOR",
                                {{"sbcc", specs.scheme, "", ""}},
                                "SBCC",
                                kernel,
                                launchers);
    }
    else if(specs.scheme == "CS_KERNEL_STOCKHAM_BLOCK_RC")
    {
        StockhamKernelRC kernel(specs);
        output += append_common_functions(kernel.generate_lds_to_reg_input_function(),
                                          kernel.generate_lds_from_reg_output_function(),
                                          {},
                                          {});
        output += make_variants(
            kernel.generate_device_function(), {}, kernel.generate_global_function(), false);

        std::vector<LaunchSuffix> suffixes;
        suffixes.push_back({"sbrc", "CS_KERNEL_STOCKHAM_BLOCK_RC", "SBRC_2D", "NONE"});
        suffixes.push_back(
            {"sbrc_unaligned", "CS_KERNEL_STOCKHAM_BLOCK_RC", "SBRC_2D", "TILE_UNALIGNED"});
        suffixes.push_back({"sbrc3d_fft_trans_xy_z_tile_aligned",
                            "CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z",
                            "SBRC_3D_FFT_TRANS_XY_Z",
                            "TILE_ALIGNED"});
        suffixes.push_back({"sbrc3d_fft_trans_xy_z_tile_unaligned",
                            "CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z",
                            "SBRC_3D_FFT_TRANS_XY_Z",
                            "TILE_UNALIGNED"});
        suffixes.push_back({"sbrc3d_fft_trans_xy_z_diagonal",
                            "CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z",
                            "SBRC_3D_FFT_TRANS_XY_Z",
                            "DIAGONAL"});
        suffixes.push_back({"sbrc3d_fft_trans_z_xy_tile_aligned",
                            "CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY",
                            "SBRC_3D_FFT_TRANS_Z_XY",
                            "TILE_ALIGNED"});
        suffixes.push_back({"sbrc3d_fft_trans_z_xy_tile_unaligned",
                            "CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY",
                            "SBRC_3D_FFT_TRANS_Z_XY",
                            "TILE_UNALIGNED"});
        suffixes.push_back({"sbrc3d_fft_trans_z_xy_diagonal",
                            "CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY",
                            "SBRC_3D_FFT_TRANS_Z_XY",
                            "DIAGONAL"});
        suffixes.push_back({"sbrc3d_fft_erc_trans_z_xy_tile_aligned",
                            "CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY",
                            "SBRC_3D_FFT_ERC_TRANS_Z_XY",
                            "TILE_ALIGNED"});
        suffixes.push_back({"sbrc3d_fft_erc_trans_z_xy_tile_unaligned",
                            "CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY",
                            "SBRC_3D_FFT_ERC_TRANS_Z_XY",
                            "TILE_UNALIGNED"});

        output += make_launcher(specs.length,
                                false,
                                specs.precisions,
                                "POWX_LARGE_SBRC_GENERATOR",
                                suffixes,
                                "SBRC",
                                kernel,
                                launchers);
    }
    else if(specs.scheme == "CS_KERNEL_STOCKHAM_BLOCK_CR")
    {
        StockhamKernelCR kernel(specs);
        output += append_common_functions(kernel.generate_lds_to_reg_input_function(),
                                          kernel.generate_lds_from_reg_output_function(),
                                          {},
                                          {});
        output += make_variants(
            kernel.generate_device_function(), {}, kernel.generate_global_function(), false);

        output += make_launcher(specs.length,
                                false,
                                specs.precisions,
                                "POWX_LARGE_SBCR_GENERATOR",
                                {{"sbcr", specs.scheme, "", ""}},
                                "SBCR",
                                kernel,
                                launchers);
    }
    else if(specs.scheme == "CS_KERNEL_2D_SINGLE")
    {
        StockhamKernelFused2D fused2d(specs, specs2d);

        auto                    device0  = fused2d.kernel0.generate_device_function();
        auto                    lds2reg0 = fused2d.kernel0.generate_lds_to_reg_input_function();
        auto                    reg2lds0 = fused2d.kernel0.generate_lds_from_reg_output_function();
        std::optional<Function> device1;
        std::optional<Function> lds2reg1;
        std::optional<Function> reg2lds1;
        if(specs.length != specs2d.length)
        {
            device1  = fused2d.kernel1.generate_device_function();
            lds2reg1 = fused2d.kernel1.generate_lds_to_reg_input_function();
            reg2lds1 = fused2d.kernel1.generate_lds_from_reg_output_function();
        }
        auto global = fused2d.generate_global_function();

        output += append_common_functions(lds2reg0, reg2lds0, lds2reg1, reg2lds1);

        output += make_variants(device0, device1, global, true);

        // output 2D launchers
        std::string length_fn
            = std::to_string(fused2d.kernel0.length) + "_" + std::to_string(fused2d.kernel1.length);
        std::string length_x = "length" + std::to_string(fused2d.kernel0.length) + "x"
                               + std::to_string(fused2d.kernel1.length);

        static const std::array<std::array<const char*, 2>, 2> precisions{
            {{"float", "sp"}, {"double", "dp"}}};

        for(auto prec_type : specs.precisions)
        {
            auto&&      prec = precisions[prec_type];
            std::string launcher
                = "rocfft_internal_dfn_" + std::string(prec[1]) + "_ci_ci_2D_" + length_fn;
            output += "POWX_SMALL_GENERATOR(" + launcher + ",ip_forward_" + length_x
                      + ",ip_inverse_" + length_x + ",op_forward_" + length_x + ",op_inverse_"
                      + length_x + "," + prec[0] + "2);";
            launchers.emplace_back(
                fused2d, specs.scheme, launcher, (prec_type == rocfft_precision_double), "", "");
        }
    }
    else
        throw std::runtime_error("unhandled scheme");

    // output json describing the launchers that were generated, so
    // kernel-generator can generate the function pool
    const char* LIST_DELIM = "";
    const char* COMMA      = ",";

    std::ofstream metadata_file((filename + ".json").c_str());

    if(!metadata_file)
        throw std::runtime_error("could not create kernel metadata file");

    metadata_file << "[";
    for(auto& launcher : launchers)
    {
        metadata_file << LIST_DELIM;
        metadata_file << launcher.to_string() << "\n";
        LIST_DELIM = COMMA;
    }
    metadata_file << "]" << std::endl;
    return output;
}
