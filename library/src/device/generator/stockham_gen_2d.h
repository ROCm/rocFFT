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

// inherits RR for convenient access to variables to generate global
// function with, but contains RR kernels for the device functions
// since they have distinct specs
struct StockhamKernelFused2D : public StockhamKernelRR
{
    StockhamKernelFused2D(StockhamGeneratorSpecs& specs0, StockhamGeneratorSpecs& specs1)
        : StockhamKernelRR(specs0)
        , kernel0(specs0)
        , kernel1(specs1)
    {
        threads_per_transform = std::max(specs1.length * specs0.threads_per_transform,
                                         specs0.length * specs1.threads_per_transform);
        // 2D_SINGLE does one 2D slab per workgroup(threadblock)
        workgroup_size       = threads_per_transform;
        transforms_per_block = 1;
        R.size               = std::max(kernel0.nregisters, kernel1.nregisters);
        kernel0.writeGuard   = true;
        kernel1.writeGuard   = true;
        // 2D kernels use kernel0 and kernel1 device functions,
        // so this writeGuard value is not used and irrelevant
        writeGuard = true;
        // // 2D_SINGLEs haven't implemented this (global function)
        // direct_to_from_reg = false;
    }

    StockhamKernelRR kernel0;
    StockhamKernelRR kernel1;

    std::vector<unsigned int> launcher_lengths() override
    {
        return {kernel0.length, kernel1.length};
    }
    std::vector<unsigned int> launcher_factors() override
    {
        std::vector<unsigned int> ret;
        std::copy(kernel0.factors.begin(), kernel0.factors.end(), std::back_inserter(ret));
        std::copy(kernel1.factors.begin(), kernel1.factors.end(), std::back_inserter(ret));
        return ret;
    }

    std::vector<Expression> device_call_arguments(unsigned int call_iter) override
    {
        return {R,
                lds_real,
                lds_complex,
                twiddles,
                stride_lds,
                call_iter ? Expression{offset_lds + call_iter * stride_lds * transforms_per_block}
                          : Expression{offset_lds},
                write};
    }

    Function generate_global_function() override
    {
        auto is_pow2 = [](unsigned int n) { return n != 0 && (n & (n - 1)) == 0; };

        auto length0 = kernel0.length;
        auto length1 = kernel1.length;

        auto length0_padded = is_pow2(length0) ? (length0 + 1) : length0;

        Function f{"forward_length" + std::to_string(length) + "x"
                   + std::to_string(kernel1.length)};

        StatementList& body = f.body;
        body += LineBreak{};
        body += CommentLines{"",
                             "this kernel:",
                             "  uses " + std::to_string(threads_per_transform)
                                 + " threads per 2d transform",
                             "  does 1 2d transforms per thread block",
                             "therefore it should be called with " + std::to_string(workgroup_size)
                                 + " threads per block",
                             ""};

        Variable d{"d", "int"};
        Variable index_along_d{"index_along_d", "size_t"};
        Variable remaining{"remaining", "size_t"};
        Variable plength{"plength", "size_t"};

        Variable batch0{"batch0", "size_t"};

        body += LDSDeclaration{scalar_type.name};
        body += Declaration{R};
        body += Declaration{thread};
        body += Declaration{transform};
        body += Declaration{offset, 0};
        body += Declaration{offset_lds};
        body += Declaration{stride_lds};
        body += Declaration{write};
        body += Declaration{batch0};
        body += Declaration{remaining};
        body += Declaration{plength, 1};
        body += Declaration{index_along_d};
        body += Declaration{lds_is_real, "false"};
        body += Declaration{lds_linear, "true"};
        body += Declaration{direct_load_to_reg, "false"};
        body += Declaration{direct_store_from_reg, "false"};
        body += CallbackDeclaration{scalar_type.name, callback_type.name};

        body += LineBreak{};
        body += CommentLines{"transform is: 2D slab number (1 per block)"};
        body += Assign{transform, block_id};
        body += Assign{remaining, transform};
        body += CommentLines{"compute 2D slab offset (start from length/stride index 2)"};

        if(static_dim)
        {
            body += Declaration{dim, static_dim};
        }
        body += For{d,
                    2,
                    d < dim,
                    1,
                    {Assign{plength, plength * lengths[d]},
                     Assign{index_along_d, remaining % lengths[d]},
                     Assign{remaining, remaining / lengths[d]},
                     Assign{offset, offset + index_along_d * stride[d]}}};
        body += Assign{batch0, transform / plength};
        body += Assign{offset, offset + batch0 * stride[dim]};

        // load
        body += LineBreak{};
        auto rw_iters = length0 * length1 / workgroup_size;
        body += CommentLines{"load length-" + std::to_string(length0) + " rows using all threads.",
                             "need " + std::to_string(rw_iters) + " iterations to load all "
                                 + std::to_string(length1) + " rows in the slab"};

        // just use rw_iters * workgroup_size threads total, break
        // it down into row/column accesses to fill LDS
        for(unsigned int i = 0; i < rw_iters; ++i)
        {
            auto row_offset = Parens{(i * workgroup_size + thread_id) / length0};
            auto col_offset = Parens{(i * workgroup_size + thread_id) % length0};
            body += Assign{
                lds_complex[row_offset * length0_padded + col_offset],
                LoadGlobal{buf, offset + col_offset * stride[0] + row_offset * stride[1]}};
        }

        body += LineBreak{};
        body += CommentLines{"", "length: " + std::to_string(length0), ""};

        body += LineBreak{};
        auto height              = kernel0.threads_per_transform;
        auto active_threads_rows = kernel0.threads_per_transform * kernel1.length;

        body += CommentLines{"each block handles " + std::to_string(length1) + " rows of length "
                                 + std::to_string(length0) + ".",
                             "each row needs " + std::to_string(kernel0.threads_per_transform)
                                 + " threads, so " + std::to_string(active_threads_rows)
                                 + " are active in the block"};

        if(active_threads_rows == workgroup_size)
            body += Assign{write, 1};
        else
            body += Assign{write, thread_id < active_threads_rows};
        body += Assign{thread, thread_id % height};
        body += Assign{offset_lds, length0_padded * (thread_id / height)};

        auto templates = device_call_templates();
        templates.set_value(stride_type.name, "SB_UNIT");
        body += Assign{stride_lds, 1};
        body += Call{"forward_length" + std::to_string(length0) + "_SBRR_device",
                     templates,
                     device_call_arguments(0)};

        // note there is a syncthreads at the start of the next call

        body += LineBreak{};
        body += CommentLines{"", "length: " + std::to_string(length1), ""};
        body += LineBreak{};

        height                   = kernel1.threads_per_transform;
        auto active_threads_cols = kernel1.threads_per_transform * length0;

        body += CommentLines{"each block handles " + std::to_string(length0) + " columns of length "
                                 + std::to_string(length1) + ".",
                             "each column needs " + std::to_string(kernel1.threads_per_transform)
                                 + " threads, so " + std::to_string(active_threads_cols)
                                 + " are active in the block"};

        if(active_threads_cols == workgroup_size)
            body += Assign{write, 1};
        else
            body += Assign{write, thread_id < active_threads_cols};
        body += Assign{thread, thread_id % height};
        body += Assign{offset_lds, thread_id / height};

        auto templates2 = device_call_templates();
        templates2.set_value(stride_type.name, "SB_NONUNIT");
        auto arguments2 = device_call_arguments(0);
        if(factors != kernel1.factors)
            arguments2[3] = twiddles + length0 - factors.front();
        body += Assign{stride_lds, length0_padded};
        body += Call{
            "forward_length" + std::to_string(length1) + "_SBRR_device", templates2, arguments2};

        // store
        body += SyncThreads{};
        body += LineBreak{};
        body += CommentLines{"store length-" + std::to_string(length0) + " rows using all threads.",
                             "need " + std::to_string(rw_iters) + " iterations to store all "
                                 + std::to_string(length1) + " rows in the slab"};

        // just use rw_iters * workgroup_size threads total, break
        // it down into row/column accesses to fill LDS
        for(unsigned int i = 0; i < rw_iters; ++i)
        {
            auto row_offset = Parens{(i * workgroup_size + thread_id) / length0};
            auto col_offset = Parens{(i * workgroup_size + thread_id) % length0};
            body += StoreGlobal{buf,
                                offset + col_offset * stride[0] + row_offset * stride[1],
                                lds_complex[row_offset * length0_padded + col_offset]};
        }

        f.qualifier     = "__global__";
        f.templates     = global_templates();
        f.arguments     = global_arguments();
        f.launch_bounds = workgroup_size;
        return f;
    }
};
