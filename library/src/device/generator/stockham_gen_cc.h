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
#include "stockham_gen_base.h"

struct StockhamKernelCC : public StockhamKernel
{
    explicit StockhamKernelCC(StockhamGeneratorSpecs& specs)
        : StockhamKernel(specs)
    {
        large_twiddle_steps.decl_default = 3;
        large_twiddle_base.decl_default  = 8;
    }

    //
    // templates
    //
    Variable apply_large_twiddle{"apply_large_twiddle", "bool"};
    Variable large_twiddle_steps{"large_twiddle_steps", "size_t"};
    Variable large_twiddle_base{"large_twiddle_base", "size_t"};

    //
    // arguments
    //
    Variable large_twiddles{"large_twiddles", "const scalar_type", true};
    Variable trans_local{"trans_local", "size_t"};

    //
    // locals
    //
    Variable tile_index{"tile_index", "size_t"};
    Variable num_of_tiles{"num_of_tiles", "size_t"};
    Variable edge{"edge", "bool"};
    Variable tid1{"tid1", "size_t"};
    Variable tid0{"tid0", "size_t"};

    // large twiddle support
    Multiply ltwd_entries{Parens{ShiftLeft{1, large_twiddle_base}}, 3};
    And      ltwd_in_lds{apply_large_twiddle, Less{large_twiddle_base, 8}};
    Variable large_twd_lds{"large_twd_lds",
                           "__shared__ scalar_type",
                           false,
                           false,
                           Ternary{Parens{ltwd_in_lds}, Parens{ltwd_entries}, Parens{0}}};

    std::string tiling_name() override
    {
        return "SBCC";
    }

    // NB:
    //   Minimize the impact of "direct_to_reg" in SBCC only for now.
    //   We should merge all back to base class eventually.
    Function generate_global_function() override
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

        if(direct_to_reg && !half_lds)
            body += Declaration{lds_linear, Literal{"false"}};
        else
            body += Declaration{lds_linear, Literal{"true"}};

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
            //FIXME: We should not control lds_linear directly. Instead, control direct_to_reg
            //body += If{And{Not{lds_is_real}, Not{lds_linear}}}, loadlds};
            if(direct_to_reg && !half_lds)
                body += If{lds_linear, loadlds};
            else
                body += If{Not{lds_is_real}, loadlds};
            body += Else{loadr};
        }

        body += LineBreak{};
        body += CommentLines{"transform"};
        for(unsigned int c = 0; c < n_device_calls; ++c)
        {
            auto templates = device_call_templates();
            auto arguments = device_call_arguments(c);

            templates.set_value(stride_type.name, "SB_UNIT");

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
            //FIXME: We should not control lds_linear directly. Instead, control direct_to_reg
            if(direct_to_reg && !half_lds)
                body += If{lds_linear, storelds};
            else
                body += If{Not{lds_is_real}, storelds};
            body += Else{storer};
        }

        f.templates = global_templates();
        f.arguments = global_arguments();
        return f;
    }

    StatementList calculate_offsets() override
    {
        Variable d{"d", "int"};
        Variable index_along_d{"index_along_d", "size_t"};
        Variable remaining{"remaining", "size_t"};
        Variable plength{"plength", "size_t"};

        StatementList stmts;
        stmts += Declaration{tile_index};
        stmts += Declaration{num_of_tiles};

        stmts += LineBreak{};
        stmts += CommentLines{"calculate offset for each tile:",
                              "  tile_index  now means index of the tile along dim1",
                              "  num_of_tiles now means number of tiles along dim1"};
        stmts += Declaration{plength, 1};
        stmts += Declaration{remaining};
        stmts += Declaration{index_along_d};
        stmts += Assign{num_of_tiles, (lengths[1] - 1) / transforms_per_block + 1};
        stmts += Assign{plength, num_of_tiles};
        stmts += Assign{tile_index, block_id % num_of_tiles};
        stmts += Assign{remaining, block_id / num_of_tiles};
        stmts += Assign{offset, tile_index * transforms_per_block * stride[1]};

        stmts += For{d,
                     2,
                     d < dim,
                     1,
                     {Assign{plength, plength * lengths[d]},
                      Assign{index_along_d, remaining % lengths[d]},
                      Assign{remaining, remaining / lengths[d]},
                      Assign{offset, offset + index_along_d * stride[d]}}};

        stmts += LineBreak{};
        if(!direct_to_reg)
        {
            stmts += Assign{transform,
                            tile_index * transforms_per_block + thread_id / threads_per_transform};
        }
        else
        {
            stmts += Assign{
                transform,
                Ternary{lds_linear,
                        tile_index * transforms_per_block + thread_id / threads_per_transform,
                        tile_index * transforms_per_block + thread_id % transforms_per_block}};
        }
        stmts += Assign{batch, block_id / plength};
        stmts += Assign{offset, offset + batch * stride[dim]};
        stmts += Assign{stride_lds, (length + get_lds_padding())};
        stmts += CommentLines{
            "TODO- currently when lds_linear is false:",
            "  offset_lds is not used in device_func, we still can sort-out the code to use it.",
            "  But EmbeddedType Pre/Post haven't supported nonlinear yet, ",
            "  so we avoid using them in the plan tree."};
        stmts += Assign{offset_lds, stride_lds * (transform % transforms_per_block)};

        stmts += Declaration{edge};
        stmts += Declaration{tid0};
        stmts += Declaration{tid1};

        stmts += Assign{
            edge,
            Ternary{Parens((tile_index + 1) * transforms_per_block > lengths[1]), "true", "false"}};

        // [dim0, dim1] = [tid0, tid1] :
        // each thread reads position [tid0, tid1], [tid0+step_h*1, tid1] , [tid0+step_h*2, tid1]...
        // tid0 walks the columns; tid1 walks the rows
        stmts += Assign{tid0, thread_id / transforms_per_block};
        stmts += Assign{tid1, thread_id % transforms_per_block};

        return stmts;
    }

    StatementList load_from_global(bool load_registers) override
    {
        auto stripmine_w = transforms_per_block;
        auto stripmine_h = workgroup_size / stripmine_w;

        StatementList stmts;

        auto offset_tile_rbuf
            = [&](unsigned int i) { return tid1 * stride[1] + (tid0 + i * stripmine_h) * stride0; };
        auto offset_tile_wlds
            = [&](unsigned int i) { return tid1 * stride_lds + (tid0 + i * stripmine_h) * 1; };

        StatementList tmp_stmts;
        Expression    pred{tile_index * transforms_per_block + tid1 < lengths[1]};
        if(!load_registers)
            for(unsigned int i = 0; i < length / stripmine_h; ++i)
                tmp_stmts += Assign{lds_complex[offset_tile_wlds(i)],
                                    LoadGlobal{buf, offset + offset_tile_rbuf(i)}};
        else
            for(unsigned int i = 0; i < length / stripmine_h; ++i)
                tmp_stmts += Assign{R[i], LoadGlobal{buf, offset + offset_tile_rbuf(i)}};

        stmts += If{Not{edge}, tmp_stmts};
        stmts += If{edge, {If{pred, tmp_stmts}}};
        return stmts;
    }

    StatementList store_to_global(bool store_registers) override
    {
        auto stripmine_w = transforms_per_block;
        auto stripmine_h = workgroup_size / stripmine_w;

        StatementList stmts;

        auto offset_tile_wbuf
            = [&](unsigned int i) { return tid1 * stride[1] + (tid0 + i * stripmine_h) * stride0; };
        auto offset_tile_rlds
            = [&](unsigned int i) { return tid1 * stride_lds + (tid0 + i * stripmine_h) * 1; };
        StatementList tmp_stmts;
        Expression    pred{tile_index * transforms_per_block + tid1 < lengths[1]};

        if(!store_registers)
            for(unsigned int i = 0; i < length / stripmine_h; ++i)
                tmp_stmts += StoreGlobal{
                    buf, offset + offset_tile_wbuf(i), lds_complex[offset_tile_rlds(i)]};
        else
            for(unsigned int i = 0; i < length / stripmine_h; ++i)
                tmp_stmts += StoreGlobal{buf, offset + offset_tile_wbuf(i), R[i]};

        stmts += If{Not{edge}, tmp_stmts};
        stmts += If{edge, {If{pred, tmp_stmts}}};
        return stmts;
    }

    TemplateList device_templates() override
    {
        TemplateList tpls = StockhamKernel::device_templates();
        tpls.append(apply_large_twiddle);
        tpls.append(large_twiddle_steps);
        tpls.append(large_twiddle_base);
        return tpls;
    }

    ArgumentList device_arguments() override
    {
        ArgumentList args = StockhamKernel::device_arguments();
        args.append(large_twiddles);
        args.append(trans_local);
        return args;
    }

    TemplateList global_templates() override
    {
        TemplateList tpls = StockhamKernel::global_templates();
        tpls.append(apply_large_twiddle);
        tpls.append(large_twiddle_steps);
        tpls.append(large_twiddle_base);
        return tpls;
    }

    ArgumentList global_arguments() override
    {
        // insert large twiddles
        ArgumentList arglist = StockhamKernel::global_arguments();
        arglist.arguments.insert(arglist.arguments.begin() + 1, large_twiddles);
        return arglist;
    }

    TemplateList device_call_templates() override
    {
        TemplateList tpls = StockhamKernel::device_call_templates();
        tpls.append(apply_large_twiddle);
        tpls.append(large_twiddle_steps);
        tpls.append(large_twiddle_base);
        return tpls;
    }

    std::vector<Expression> device_call_arguments(unsigned int call_iter) override
    {
        std::vector<Expression> args = StockhamKernel::device_call_arguments(call_iter);
        auto which = Ternary{Parens{And{apply_large_twiddle, large_twiddle_base < 8}},
                             Parens{large_twd_lds},
                             Parens{large_twiddles}};
        args.push_back(which);
        args.push_back(transform);
        return args;
    }

    StatementList large_twiddles_load() override
    {
        Variable ltwd_id{"ltwd_id", "size_t"};

        StatementList stmts;
        stmts += Declaration{large_twd_lds};
        stmts += If{ltwd_in_lds,
                    {Declaration{ltwd_id, thread_id},
                     While{Less{ltwd_id, ltwd_entries},
                           {Assign{large_twd_lds[ltwd_id], large_twiddles[ltwd_id]},
                            AddAssign(ltwd_id, workgroup_size)}}}};

        return stmts;
    }

    StatementList large_twiddles_multiply(unsigned int width, unsigned int cumheight) override
    {
        StatementList stmts;
        stmts += CommentLines{"large twiddle multiplication"};
        for(unsigned int w = 0; w < width; ++w)
        {
            // FIXME- using a .cast('type') would be graceful!
            //        Why casting to ((int)thread % 8) only when passing to TW_NSteps ?
            //        This is completely based on the testing result. We observed that it can
            //        reduce a few vgprs (we don't know why since its behind the compiler)
            //        and avoid the drop of occuapancy (espcially for sbcc_len64_inverse)
            // idx = Parens(Parens(thread % cumheight) + w * cumheight) * trans_local
            auto idx = std::string("(((int)") + thread.render() + " % " + std::to_string(cumheight)
                       + ") + " + std::to_string(w) + " * " + std::to_string(cumheight) + ") * "
                       + trans_local.render();
            stmts += Assign{
                W,
                CallExpr{"TW_NSteps",
                         TemplateList{scalar_type, large_twiddle_base, large_twiddle_steps},
                         {large_twiddles, idx}}};
            stmts += Assign{t, TwiddleMultiply{R[w], W}};
            stmts += Assign{R[w], t};
        }
        return {If{apply_large_twiddle, stmts}};
    }
};
