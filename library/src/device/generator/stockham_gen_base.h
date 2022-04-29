// Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once
#include "stockham_gen.h"

#include <cmath>

// Base class for stockham kernels.  Subclasses are responsible for
// different tiling types.
//
// This inherits from specs as a shortcut to avoid having to look
// inside a class member for variables we need all the time
struct StockhamKernel : public StockhamGeneratorSpecs
{
    // Currently, we aim for minimum occupancy of 2 for these
    // kernels.  Assuming current hardware has 64kiB of LDS, that
    // limits our kernels to 32 kiB.
    //
    // This byte limit is a constant now, but could be turned into an
    // input parameter or be made changeable by derived classes.
    static const unsigned int LDS_BYTE_LIMIT    = 32 * 1024;
    static const unsigned int BYTES_PER_ELEMENT = 16;
    StockhamKernel(StockhamGeneratorSpecs& specs)
        : StockhamGeneratorSpecs(specs)
    {
        auto bytes_per_batch = length * BYTES_PER_ELEMENT;

        if(half_lds)
            bytes_per_batch /= 2;

        if(threads_per_transform == 0)
        {
            threads_per_transform = 1;
            for(unsigned int t = 2; t < length; ++t)
            {
                if(t > workgroup_size)
                    continue;
                if(length % t == 0)
                {
                    if(std::all_of(factors.begin(), factors.end(), [=](unsigned int f) {
                           return (length / t) % f == 0;
                       }))
                        threads_per_transform = t;
                }
            }
        }

        transforms_per_block = LDS_BYTE_LIMIT / bytes_per_batch;
        while(threads_per_transform * transforms_per_block > workgroup_size)
            --transforms_per_block;
        if(!factors2d.empty())
            transforms_per_block = std::min(transforms_per_block, length2d);

        workgroup_size = threads_per_transform * transforms_per_block;
        nregisters     = compute_nregisters(length, factors, threads_per_transform);
        R.size         = Expression{nregisters};
    }
    virtual ~StockhamKernel(){};

    unsigned int nregisters;
    unsigned int transforms_per_block;

    // data that may be overridden by subclasses (different tiling types)
    unsigned int n_device_calls = 1;
    bool         writeGuard     = false;

    static unsigned int compute_nregisters(unsigned int                     length,
                                           const std::vector<unsigned int>& factors,
                                           unsigned int                     threads_per_transform)
    {
        unsigned int max_registers = 0;
        for(auto width : factors)
        {
            unsigned int n = std::ceil(double(length) / width / threads_per_transform) * width;
            if(n > max_registers)
                max_registers = n;
        }
        return max_registers;
    }

    //
    // templates
    //
    Variable scalar_type{"scalar_type", "typename"};
    Variable callback_type{"cbtype", "CallbackType"};
    Variable stride_type{"sb", "StrideBin"};
    Variable embedded_type{"ebtype", "EmbeddedType"};
    Variable directReg_type{"drtype", "DirectRegType"};

    //
    // arguments
    //
    // global input/ouput buffer
    Variable buf{"buf", "scalar_type", true, true};

    // global twiddle table (stacked)
    Variable twiddles{"twiddles", "const scalar_type", true, true};

    // rank/dimension of transform
    Variable dim{"dim", "const size_t"};

    // transform lengths
    Variable lengths{"lengths", "const size_t", true, true};

    // input/output array strides
    Variable stride{"stride", "const size_t", true, true};

    // number of transforms/batches
    Variable nbatch{"nbatch", "const size_t"};

    // the number of padding at the end of each row in lds
    Variable lds_padding{"lds_padding", "const unsigned int"};

    // should the device function write to lds?
    // only used for 2D
    Variable write{"write", "bool"};

    // is LDS real-only?
    Variable lds_is_real = Variable{"lds_is_real", "const bool"};

    // The LDS access pattern is linear both on R/W
    Variable lds_linear = Variable{"lds_linear", "const bool"};

    // Enable directly load to registers and directly store from registers?
    // This variable affects the lds_linear (internally)
    Variable direct_to_from_reg = Variable{"direct_to_from_reg", "const bool"};

    //
    // locals
    //
    // lds storage buffer
    Variable lds_real{"lds_real", "real_type_t<scalar_type>", true, true};
    Variable lds_complex{"lds_complex", "scalar_type", true, true};
    Variable lds_row_padding{"lds_row_padding", "unsigned int"};

    // hip thread block id
    Variable block_id{"blockIdx.x", "unsigned int"};

    // hip thread id
    Variable thread_id{"threadIdx.x", "unsigned int"};

    // thread within transform
    Variable thread{"thread", "size_t"};

    // global input/output buffer offset to current transform
    Variable offset{"offset", "size_t"};

    // lds buffer offset to current transform
    Variable offset_lds{"offset_lds", "unsigned int"};

    // current batch
    Variable batch{"batch", "size_t"};

    // current transform index in a batch
    Variable transform{"transform", "size_t"};

    // stride between consecutive indexes
    Variable stride0{"stride0", "const size_t"};

    // stride between consecutive indexes in lds
    Variable stride_lds{"stride_lds", "size_t"};

    // usually in device: const size_t lstride = (sb == SB_UNIT) ? 1 : stride_lds;
    // with this definition, the compiler knows that "index * lstride" is trivial under SB_UNIT
    Variable lstride{"lstride", "const size_t"};

    // twiddle value during twiddle application
    Variable W{"W", "scalar_type"};

    // temporary register during twiddle application
    Variable t{"t", "scalar_type"};

    // butterfly registers
    Variable R{"R", "scalar_type", false, false};

    virtual std::vector<unsigned int> launcher_lengths()
    {
        return {length};
    }
    virtual std::vector<unsigned int> launcher_factors()
    {
        return factors;
    }

    virtual TemplateList device_templates()
    {
        TemplateList tpls;
        tpls.append(scalar_type);
        tpls.append(lds_is_real);
        tpls.append(stride_type);
        tpls.append(lds_linear);
        return tpls;
    }

    virtual TemplateList global_templates()
    {
        return {scalar_type, stride_type, embedded_type, callback_type, directReg_type};
    }

    virtual ArgumentList device_arguments()
    {
        ArgumentList args{R, lds_real, lds_complex, twiddles, stride_lds, offset_lds, write};
        return args;
    }

    virtual ArgumentList global_arguments()
    {
        auto arguments = static_dim
                             ? ArgumentList{twiddles, lengths, stride, nbatch, lds_padding}
                             : ArgumentList{twiddles, dim, lengths, stride, nbatch, lds_padding};
        for(const auto& arg : get_callback_args().arguments)
            arguments.append(arg);
        arguments.append(buf);
        return arguments;
    }

    // TODO- need to avoid the involvement of half-lds/lds_is_real
    virtual StatementList set_direct_to_from_registers()
    {
        if(direct_to_reg)
            return {Declaration{direct_to_from_reg, lds_is_real},
                    Declaration{lds_linear, Literal{"true"}}};
        else
            return {Declaration{direct_to_from_reg, Literal{"false"}},
                    Declaration{lds_linear, Literal{"true"}}};
    }

    virtual StatementList large_twiddles_load()
    {
        return {CommentLines{"- no large twiddles"}};
    }

    virtual StatementList large_twiddles_multiply(unsigned int width, unsigned int cumheight)
    {
        return {};
    }

    virtual StatementList check_batch()
    {
        return {If{batch >= nbatch, {Return{}}}};
    }

    static ArgumentList get_callback_args()
    {
        return {Variable{"load_cb_fn", "void", true, true},
                Variable{"load_cb_data", "void", true, true},
                Variable{"load_cb_lds_bytes", "uint32_t"},
                Variable{"store_cb_fn", "void", true, true},
                Variable{"store_cb_data", "void", true, true}};
    }

    // we currently only use LDS padding for embedded R2C/C2R, so
    // there's no reason to look at the lds_padding parameter
    // otherwise.
    virtual Expression get_lds_padding()
    {
        return Ternary{embedded_type == "EmbeddedType::NONE", 0, lds_padding};
    }

    // load registers R from lds_complex
    enum class Component
    {
        NONE,
        X,
        Y,
    };
    StatementList load_lds_generator(
        unsigned int h, unsigned int hr, unsigned int width, unsigned int dt, Component component)
    {
        if(hr == 0)
            hr = h;
        StatementList load;
        for(unsigned int w = 0; w < width; ++w)
        {
            const auto tid = Parens{thread + dt + h * threads_per_transform};
            const auto idx = offset_lds + (tid + w * length / width) * lstride;
            switch(component)
            {
            case Component::X:
                load += Assign(R[hr * width + w].x, lds_real[idx]);
                break;
            case Component::Y:
                load += Assign(R[hr * width + w].y, lds_real[idx]);
                break;
            case Component::NONE:
                load += Assign(R[hr * width + w], lds_complex[idx]);
                break;
            }
        }
        return load;
    }

    StatementList store_lds_generator(unsigned int h,
                                      unsigned int hr,
                                      unsigned int width,
                                      unsigned int dt,
                                      Component    component,
                                      unsigned int cumheight)
    {
        if(hr == 0)
            hr = h;
        StatementList work;
        for(unsigned int w = 0; w < width; ++w)
        {
            const auto tid = thread + dt + h * threads_per_transform;
            const auto idx = offset_lds
                             + (Parens{tid / cumheight} * (width * cumheight) + tid % cumheight
                                + w * cumheight)
                                   * lstride;
            switch(component)
            {
            case Component::X:
                work += Assign(lds_real[idx], R[hr * width + w].x);
                break;
            case Component::Y:
                work += Assign(lds_real[idx], R[hr * width + w].y);
                break;
            case Component::NONE:
                work += Assign(lds_complex[idx], R[hr * width + w]);
                break;
            }
        }
        return work;
    }

    StatementList apply_twiddle_generator(unsigned int h,
                                          unsigned int hr,
                                          unsigned int width,
                                          unsigned int dt,
                                          unsigned int cumheight)
    {
        if(hr == 0)
            hr = h;
        StatementList work;
        for(unsigned int w = 1; w < width; ++w)
        {
            auto tid  = thread + dt + h * threads_per_transform;
            auto tidx = cumheight - 1 + w - 1 + (width - 1) * (tid % cumheight);
            auto ridx = hr * width + w;
            work += Assign(W, twiddles[tidx]);
            work += Assign(t, TwiddleMultiply(R[ridx], W));
            work += Assign(R[ridx], t);
        }
        return work;
    }

    StatementList
        butterfly_generator(unsigned int h, unsigned int hr, unsigned int width, unsigned int dt)
    {
        if(hr == 0)
            hr = h;
        std::vector<Expression> args;
        for(unsigned int w = 0; w < width; ++w)
            args.push_back(R + (hr * width + w));
        return {Butterfly{true, args}};
    }

    StatementList load_global_generator(unsigned int h,
                                        unsigned int hr,
                                        unsigned int width,
                                        unsigned int dt) const
    {
        if(hr == 0)
            hr = h;
        StatementList load;
        for(unsigned int w = 0; w < width; ++w)
        {
            auto tid = Parens{thread + dt + h * threads_per_transform};
            auto idx = Parens{tid + w * length / width};
            load += Assign{R[hr * width + w],
                           LoadGlobal{buf, offset + Parens{Expression{idx}} * stride0}};
        }
        return load;
    }

    StatementList store_global_generator(unsigned int h,
                                         unsigned int hr,
                                         unsigned int width,
                                         unsigned int dt,
                                         unsigned int cumheight)
    {
        if(hr == 0)
            hr = h;
        StatementList work;
        for(unsigned int w = 0; w < width; ++w)
        {
            auto tid = thread + dt + h * threads_per_transform;
            auto idx = offset
                       + (Parens{tid / cumheight} * (width * cumheight) + tid % cumheight
                          + w * cumheight)
                             * stride0;
            work += StoreGlobal(buf, idx, R[hr * width + w]);
        }
        return work;
    }

    Function generate_device_function()
    {
        std::string function_name
            = "forward_length" + std::to_string(length) + "_" + tiling_name() + "_device";

        Function f{function_name};
        f.arguments = device_arguments();
        f.templates = device_templates();
        f.qualifier = "__device__";
        if(length == 1)
        {
            return f;
        }

        StatementList& body = f.body;
        body += Declaration{thread};
        body += Declaration{W};
        body += Declaration{t};
        body += Declaration{
            lstride, Ternary{Parens{stride_type == "SB_UNIT"}, Parens{1}, Parens{stride_lds}}};

        body += Assign{thread,
                       Ternary{lds_linear,
                               thread_id % threads_per_transform,
                               thread_id / transforms_per_block}};

        for(unsigned int npass = 0; npass < factors.size(); ++npass)
        {
            // width is the butterfly width, Radix-n. Mostly is used as dt in add_work()
            unsigned int width = factors[npass];
            // height is how many butterflies per thread will do on average
            float height = static_cast<float>(length) / width / threads_per_transform;

            unsigned int cumheight
                = product(factors.begin(),
                          factors.begin() + npass); // cumheight is irrelevant to the above height,
            // is used for twiddle multiplication and lds writing.

            body += LineBreak{};
            body += CommentLines{
                "pass " + std::to_string(npass) + ", width " + std::to_string(width),
                "using " + std::to_string(threads_per_transform) + " threads we need to do "
                    + std::to_string(length / width) + " radix-" + std::to_string(width)
                    + " butterflies",
                "therefore each thread will do " + std::to_string(height) + " butterflies"};

            auto load_lds  = std::mem_fn(&StockhamKernel::load_lds_generator);
            auto store_lds = std::mem_fn(&StockhamKernel::store_lds_generator);

            // first pass of linear variant load (full)
            // TODO- move this "first pass" part out of device function.
            if(npass == 0)
            {
                StatementList first_load_full;
                first_load_full += SyncThreads();
                first_load_full
                    += add_work(std::bind(load_lds, this, _1, _2, _3, _4, Component::NONE),
                                width,
                                height,
                                false);
                body += If{lds_linear, {If{Not{lds_is_real}, first_load_full}}};
            }

            if(npass > 0)
            {
                // internal full lds load (both linear/nonlinear variants)
                StatementList internal_load_full;
                internal_load_full += SyncThreads();
                internal_load_full
                    += add_work(std::bind(load_lds, this, _1, _2, _3, _4, Component::NONE),
                                width,
                                height,
                                false);
                body += If{Not{lds_is_real}, internal_load_full};

                auto apply_twiddle = std::mem_fn(&StockhamKernel::apply_twiddle_generator);
                body += add_work(std::bind(apply_twiddle, this, _1, _2, _3, _4, cumheight),
                                 width,
                                 height,
                                 false);
            }

            auto butterfly = std::mem_fn(&StockhamKernel::butterfly_generator);
            body += add_work(std::bind(butterfly, this, _1, _2, _3, _4), width, height, false);

            if(npass == factors.size() - 1)
                body += large_twiddles_multiply(width, cumheight);

            // internal lds store (half-with-linear and full-with-linear/nonlinear)
            StatementList store_full;
            StatementList store_half;
            if(npass < factors.size() - 1)
            {
                // linear variant store (half) and load (half)
                for(auto component : {Component::X, Component::Y})
                {
                    bool isFirstStore = (npass == 0) && (component == Component::X);
                    auto half_width   = factors[npass];
                    auto half_height
                        = static_cast<float>(length) / half_width / threads_per_transform;
                    // minimize sync as possible
                    if(!isFirstStore)
                        store_half += SyncThreads();
                    store_half += add_work(
                        std::bind(store_lds, this, _1, _2, _3, _4, component, cumheight),
                        half_width,
                        half_height,
                        true);

                    half_width  = factors[npass + 1];
                    half_height = static_cast<float>(length) / half_width / threads_per_transform;
                    store_half += SyncThreads();
                    store_half += add_work(std::bind(load_lds, this, _1, _2, _3, _4, component),
                                           half_width,
                                           half_height,
                                           true);
                }

                // internal full lds store (both linear/nonlinear variants)
                if(npass == 0)
                    store_full += If{lds_linear, {SyncThreads()}};
                else
                    store_full += SyncThreads();
                store_full += add_work(
                    std::bind(store_lds, this, _1, _2, _3, _4, Component::NONE, cumheight),
                    width,
                    height,
                    true);

                body += If{Not{lds_is_real}, store_full};
                body += Else{store_half};
            }

            // last pass of linear variant store (full)
            // TODO- move this "last pass" part out of device function.
            if(npass == factors.size() - 1)
            {
                StatementList last_store_full;
                last_store_full += SyncThreads();
                last_store_full += add_work(
                    std::bind(store_lds, this, _1, _2, _3, _4, Component::NONE, cumheight),
                    width,
                    height,
                    true);

                body += If{lds_linear, {If{Not{lds_is_real}, last_store_full}}};
            }
        }
        return f;
    }

    void collect_length_stride(StatementList& body)
    {
        if(static_dim)
        {
            body += Declaration{dim, static_dim};
        }
        body += Declaration{
            stride0, Ternary{Parens{stride_type == "SB_UNIT"}, Parens{1}, Parens{stride[0]}}};
    }

    virtual Function generate_global_function()
    {
        Function f("forward_length" + std::to_string(length) + "_" + tiling_name());
        f.qualifier     = "__global__";
        f.launch_bounds = workgroup_size;

        StatementList& body = f.body;
        body += CommentLines{
            "this kernel:",
            "  uses " + std::to_string(threads_per_transform) + " threads per transform",
            "  does " + std::to_string(transforms_per_block) + " transforms per thread block",
            "therefore it should be called with " + std::to_string(workgroup_size)
                + " threads per thread block"};
        body += Declaration{R};
        body += LDSDeclaration{scalar_type.name};
        body += Declaration{offset, 0};
        body += Declaration{offset_lds};
        body += Declaration{stride_lds};
        body += Declaration{batch};
        body += Declaration{transform};

        if(half_lds)
            body += Declaration{lds_is_real, embedded_type == "EmbeddedType::NONE"};
        else
            body += Declaration{lds_is_real, Literal{"false"}};

        // TODO- don't override, unify them
        body += set_direct_to_from_registers();

        body += CallbackDeclaration{scalar_type.name, callback_type.name};

        body += LineBreak{};
        body += CommentLines{"large twiddles"};
        body += large_twiddles_load();

        body += LineBreak{};
        body += CommentLines{"offsets"};
        collect_length_stride(body);
        body += calculate_offsets();

        body += LineBreak{};
        body += check_batch();
        body += LineBreak{};

        StatementList loadlds;
        loadlds += CommentLines{"load global into lds"};
        loadlds += load_from_global(false);
        loadlds += LineBreak{};
        loadlds += CommentLines{
            "handle even-length real to complex pre-process in lds before transform"};
        loadlds += real2cmplx_pre_post(length, ProcessingType::PRE);

        if(!direct_to_reg)
            body += loadlds;
        else
        {
            StatementList loadr;
            loadr += CommentLines{"load global into registers"};
            loadr += load_from_global(true);

            body += {If{direct_to_from_reg, loadr}, Else{loadlds}};
        }

        body += LineBreak{};
        body += CommentLines{"transform"};
        for(unsigned int c = 0; c < n_device_calls; ++c)
        {
            auto templates = device_call_templates();
            auto arguments = device_call_arguments(c);

            templates.set_value(stride_type.name, "lds_linear ? SB_UNIT : SB_NONUNIT");

            body
                += Call{"forward_length" + std::to_string(length) + "_" + tiling_name() + "_device",
                        templates,
                        arguments};
        }

        StatementList storelds;
        storelds += LineBreak{};
        storelds += CommentLines{
            "handle even-length complex to real post-process in lds after transform"};
        storelds += real2cmplx_pre_post(length, ProcessingType::POST);

        storelds += LineBreak{};

        storelds += CommentLines{"store global"};
        storelds += SyncThreads{};
        storelds += store_to_global(false);

        if(!direct_to_reg)
            body += storelds;
        else
        {
            StatementList storer;
            storer += CommentLines{"store registers into global"};
            storer += store_to_global(true);

            body += {If{direct_to_from_reg, storer}, Else{storelds}};
        }

        f.templates = global_templates();
        f.arguments = global_arguments();
        return f;
    }

    // virtual functions implemented by different tiling implementations
    virtual std::string tiling_name() = 0;

    // void update_kernel_settings();

    virtual StatementList calculate_offsets() = 0;

    virtual StatementList load_from_global(bool load_registers) = 0;

    virtual StatementList store_to_global(bool store_registers) = 0;

    virtual TemplateList device_call_templates()
    {
        return {scalar_type, lds_is_real, stride_type, lds_linear};
    }

    virtual std::vector<Expression> device_call_arguments(unsigned int call_iter)
    {
        return {R,
                lds_real,
                lds_complex,
                twiddles,
                stride_lds,
                call_iter ? Expression{offset_lds + call_iter * stride_lds * transforms_per_block}
                          : Expression{offset_lds},
                Literal{"true"}};
    }

    enum class ProcessingType
    {
        PRE,
        POST,
    };
    StatementList real2cmplx_pre_post(unsigned int half_N, ProcessingType type)
    {
        std::string function_name = type == ProcessingType::PRE
                                        ? "real_pre_process_kernel_inplace"
                                        : "real_post_process_kernel_inplace";
        std::string template_type = type == ProcessingType::PRE ? "EmbeddedType::C2Real_PRE"
                                                                : "EmbeddedType::Real2C_POST";
        Variable    Ndiv4{half_N % 2 == 0 ? "true" : "false", "bool"};
        auto        quarter_N = half_N / 2;
        if(half_N % 2 == 1)
            quarter_N += 1;

        StatementList stmts;
        // Todo: We might not have to sync here which depends on the access pattern
        stmts += SyncThreads{};
        stmts += LineBreak{};

        // Todo: For case threads_per_transform == quarter_N, we
        // could save one more "if" in the c2r/r2r kernels

        // if we have fewer threads per transform than quarter_N,
        // we need to call the pre/post function multiple times
        auto r2c_calls_per_transform = quarter_N / threads_per_transform;
        if(quarter_N % threads_per_transform > 0)
            r2c_calls_per_transform += 1;
        for(unsigned int i = 0; i < r2c_calls_per_transform; ++i)
        {
            TemplateList tpls;
            tpls.append(scalar_type);
            tpls.append(Ndiv4);
            std::vector<Expression> args{
                thread_id % threads_per_transform + i * threads_per_transform,
                half_N - thread_id % threads_per_transform - i * threads_per_transform,
                quarter_N,
                lds_complex + offset_lds,
                0,
                twiddles + half_N};
            stmts += Call{function_name, tpls, args};
        }
        if(type == ProcessingType::PRE)
        {
            stmts += SyncThreads();
            stmts += LineBreak();
        }

        return {If{Equal{embedded_type, template_type}, stmts}};
    }

    // Call generator as many times as needed.
    // generator accepts h, hr, width, dt parameters
    StatementList add_work(
        std::function<StatementList(unsigned int, unsigned int, unsigned int, unsigned int)>
                     generator,
        unsigned int width,
        double       height,
        bool         guard,
        bool         trans_dir = false) const
    {
        StatementList stmts;
        unsigned int  iheight = std::floor(height);
        if(height > iheight && threads_per_transform > length / width)
            iheight += 1;

        StatementList work;
        for(unsigned int h = 0; h < iheight; ++h)
            work += generator(h, 0, width, 0);

        if(guard)
        {
            if((!trans_dir && threads_per_transform != length / width)
               || (trans_dir && workgroup_size / transforms_per_block > length / width))
            {
                stmts += CommentLines{"more than enough threads, some do nothing"};
                if(writeGuard)
                    stmts += If{write && (thread < length / width), work};
                else
                    stmts += If{thread < length / width, work};
            }
            else
            {
                if(writeGuard)
                    stmts += If{write, work};
                else
                    stmts += work;
            }
        }
        else
        {
            stmts += work;
        }

        if(height > iheight && threads_per_transform < length / width)
        {
            stmts += CommentLines{"not enough threads, some threads do extra work"};
            unsigned int dt = iheight * threads_per_transform;
            work            = generator(0, iheight, width, dt);
            if(writeGuard)
                stmts += If{write && (thread + dt < length / width), work};
            else
                stmts += If{thread + dt < length / width, work};
        }

        return stmts;
    }
};
