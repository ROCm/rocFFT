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

struct StockhamKernelRC : public StockhamKernel
{
    explicit StockhamKernelRC(StockhamGeneratorSpecs& specs)
        : StockhamKernel(specs)
    {
        n_device_calls = block_width / transforms_per_block;
    }

    //
    // templates
    //
    Variable sbrc_type{"sbrc_type", "SBRC_TYPE"};
    Variable transpose_type{"transpose_type", "SBRC_TRANSPOSE_TYPE"};

    //
    // locals
    //
    Variable tile{"tile", "unsigned int"};
    Variable offset_in{"offset_in", "unsigned int"};
    Variable offset_out{"offset_out", "unsigned int"};
    Variable stride_in{"stride_in", "const size_t", true};
    Variable stride_out{"stride_out", "const size_t", true};

    //
    // locals
    //
    Variable tile_index{"tile_index", "size_t"};
    Variable num_of_tiles{"num_of_tiles", "size_t"};
    Variable edge{"edge", "bool"};
    Variable tid1{"tid1", "size_t"};
    Variable tid0{"tid0", "size_t"};

    std::string tiling_name() override
    {
        return "SBRC";
    }

    StatementList generate_2d_global_function()
    {
        StatementList body;

        body += LineBreak{};
        body += CommentLines{"offsets"};
        body += calculate_offsets_2d();

        body += LineBreak{};
        body += If{batch >= nbatch, {Return{}}};
        body += LineBreak{};

        StatementList loadlds;
        loadlds += CommentLines{"load global into lds"};
        loadlds += load_from_global_2d(false);
        // loadlds += LineBreak{};
        // loadlds += CommentLines{
        //     "handle even-length real to complex pre-process in lds before transform"};
        // loadlds += real2cmplx_pre_post(length, ProcessingType::PRE);

        if(load_from_lds)
            body += loadlds;
        else
        {
            StatementList loadr;
            loadr += CommentLines{"load global into registers"};
            loadr += load_from_global_2d(true);
            body += If{Not{lds_is_real}, loadlds};
            body += Else{loadr};
        }

        body += LineBreak{};
        body += CommentLines{"transform"};
        auto templates = device_call_templates();
        auto arguments = device_call_arguments(0);

        templates.set_value(stride_type.name, "SB_UNIT");
        body += Call{"forward_length" + std::to_string(length) + "_" + tiling_name() + "_device",
                     templates,
                     arguments};

        StatementList storelds;
        // storelds += LineBreak{};
        // storelds += CommentLines{
        //     "handle even-length complex to real post-process in lds after transform"};
        // storelds += real2cmplx_pre_post(length, ProcessingType::POST);

        storelds += LineBreak{};

        storelds += CommentLines{"store global"};
        storelds += SyncThreads{};
        storelds += store_to_global_2d(false);

        if(load_from_lds)
            body += storelds;
        else
        {
            StatementList storer;
            storer += CommentLines{"store registers into global"};
            storer += store_to_global_2d(true);
            body += If{Not{lds_is_real}, storelds};
            body += Else{storer};
        }

        return body;
    }

    StatementList generate_non_2d_global_function()
    {
        StatementList body;

        body += LineBreak{};
        body += CommentLines{"offsets"};
        body += calculate_offsets();

        body += LineBreak{};
        body += If{batch >= nbatch, {Return{}}};
        body += LineBreak{};

        StatementList loadlds;
        loadlds += CommentLines{"load global into lds"};
        loadlds += load_from_global(false);
        loadlds += LineBreak{};
        loadlds += CommentLines{
            "handle even-length real to complex pre-process in lds before transform"};
        loadlds += real2cmplx_pre_post(length, ProcessingType::PRE);

        if(load_from_lds)
            body += loadlds;
        else
        {
            StatementList loadr;
            loadr += CommentLines{"load global into registers"};
            loadr += load_from_global(true);
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

        if(load_from_lds)
            body += storelds;
        else
        {
            StatementList storer;
            storer += CommentLines{"store registers into global"};
            storer += store_to_global(true);
            body += If{Not{lds_is_real}, storelds};
            body += Else{storer};
        }

        return body;
    }

    Function generate_global_function() override
    {
        Function f("forward_length" + std::to_string(length) + "_" + tiling_name());
        f.qualifier     = "__global__";
        f.launch_bounds = threads_per_block;

        StatementList& body = f.body;
        body += CommentLines{
            "this kernel:",
            "  uses " + std::to_string(threads_per_transform) + " threads per transform",
            "  does " + std::to_string(transforms_per_block) + " transforms per thread block",
            "therefore it should be called with " + std::to_string(threads_per_block)
                + " threads per thread block"};
        body += Declaration{R};
        body += LDSDeclaration{scalar_type.name};
        body += Declaration{offset, 0};
        body += Declaration{offset_lds};
        body += Declaration{stride_lds};
        body += Declaration{batch};
        body += Declaration{transform};
        body += Declaration{thread};

        if(half_lds)
            body += Declaration{lds_is_real, embedded_type == "EmbeddedType::NONE"};
        else
            body += Declaration{lds_is_real, Literal{"false"}};
        body += Declaration{
            stride0, Ternary{Parens{stride_type == "SB_UNIT"}, Parens{1}, Parens{stride[0]}}};
        body += CallbackDeclaration{scalar_type.name, callback_type.name};

        body += LineBreak{};
        body += CommentLines{"large twiddles"};
        body += large_twiddles_load();

        body += If{sbrc_type == "SBRC_2D", generate_2d_global_function()};
        body += Else{generate_non_2d_global_function()};

        f.templates = global_templates();
        f.arguments = global_arguments();
        return f;
    }

    StatementList calculate_offsets() override
    {
        StatementList stmts;

        stmts += Declaration{tile};

        stmts += LineBreak{};
        stmts += CommentLines{"so far, lds_padding is either 0 or 1, assigning only 0 or 1",
                              "to a local var can make compiler able to optimize the asm,",
                              "such as removing un-neccessary add, mul and s_load, especially"
                              "when this variable is used many times!"
                              "the padding here is ROW_PADDING"};
        stmts += Declaration{lds_row_padding, 0};

        stmts += If{sbrc_type == "SBRC_3D_FFT_ERC_TRANS_Z_XY", {Assign{lds_row_padding, 1}}};
        stmts += Assign{tile, block_id};
        stmts += Assign{thread, thread_id};

        stmts += If{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z", calculate_offsets_fft_trans_xy_z()};
        stmts += If{
            Or{sbrc_type == "SBRC_3D_FFT_TRANS_Z_XY", sbrc_type == "SBRC_3D_FFT_ERC_TRANS_Z_XY"},
            calculate_offsets_fft_trans_z_xy()};

        // XXX
        stmts += Assign{batch, 0};
        return stmts;
    }

    StatementList calculate_offsets_2d()
    {
        Variable d{"d", "int"};
        Variable index_along_d{"index_along_d", "size_t"};
        Variable remaining{"remaining", "size_t"};
        Variable plength{"plength", "size_t"};

        StatementList stmts;

        stmts += Declaration{tile_index};
        stmts += Declaration{num_of_tiles};

        stmts += LineBreak{};
        stmts += Declaration{lds_row_padding, 0};

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
        stmts += Assign{transform,
                        tile_index * transforms_per_block + thread_id / threads_per_transform};
        stmts += Assign{batch, block_id / plength};
        stmts += Assign{offset, offset + batch * stride[dim]};
        stmts += Assign{stride_lds, (length + lds_row_padding)};
        stmts += Assign{offset_lds, stride_lds * (transform % transforms_per_block)};

        return stmts;
    }

    StatementList calculate_offsets_fft_trans_xy_z()
    {
        auto tiles_per_batch = Parens(lengths[1] * ((lengths[2] + block_width - 1) / block_width));
        auto threads_per_row = threads_per_block / block_width;

        auto diagonal = [&]() -> StatementList {
            Variable readTileIdx_x{"readTileIdx_x", "unsigned int"}; //, value=}
            Variable readTileIdx_y{"readTileIdx_y", "unsigned int"}; //, value=}
            Variable bid{"bid", "unsigned int"}; //, value=}
            Variable tileBlockIdx_y{"tileBlockIdx_y", "unsigned int"}; //, value=}
            Variable tileBlockIdx_x{
                "tileBlockIdx_x",
                "unsigned int"}; //, value=B{B{bid / threads_per_row} + tileBlockIdx_y} % length}
            return {
                Declaration{readTileIdx_x, tile % lengths[1]},
                Declaration{readTileIdx_y, tile % tiles_per_batch / lengths[1]},
                Declaration{bid, readTileIdx_x + length * readTileIdx_y},
                Declaration{tileBlockIdx_y, bid % threads_per_row},
                Declaration{tileBlockIdx_x,
                            (Parens{bid / threads_per_row} + tileBlockIdx_y) % length},

                AddAssign(offset_in, tileBlockIdx_x * stride_in[1]),
                AddAssign(offset_in, tileBlockIdx_y * block_width * stride_in[2]),
                AddAssign(offset_in, Parens{tile / tiles_per_batch} * stride_in[3]),
                AddAssign(offset_out, tileBlockIdx_y * block_width * stride_out[0]),
                AddAssign(offset_out, tileBlockIdx_x * stride_out[2]),
                AddAssign(offset_out, tile / tiles_per_batch * stride_out[3]),
                Assign{offset_lds, Parens{thread / threads_per_transform} * length},
            };
        };

        auto not_diagonal = [&]() -> StatementList {
            auto read_tile_x  = Parens{tile % lengths[1]};
            auto read_tile_y  = Parens{Parens{tile % tiles_per_batch} / lengths[1]};
            auto write_tile_x = read_tile_y;
            auto write_tile_y = read_tile_x;
            return {AddAssign(offset_in, read_tile_x * stride_in[1]),
                    AddAssign(offset_in, read_tile_y * block_width * stride_in[2]),
                    AddAssign(offset_in, Parens{tile / tiles_per_batch} * stride_in[3]),
                    AddAssign(offset_out, write_tile_x * block_width * stride_out[0]),
                    AddAssign(offset_out, write_tile_y * stride_out[2]),
                    AddAssign(offset_out, Parens{tile / tiles_per_batch} * stride_out[3]),
                    Assign{offset_lds, Parens{thread / threads_per_transform} * length}};
        };
        return {
            If{transpose_type == "DIAGONAL", diagonal()},
            Else{not_diagonal()},
        };
    }

    StatementList calculate_offsets_fft_trans_z_xy()
    {
        Variable tile_size_x{"tgs_x", "unsigned int"};
        Variable tile_size_y{"tgs_y", "unsigned int"};
        Variable tiles_per_batch{"tiles_per_batch", "unsigned int"};

        Variable read_tile_x{"readTileIdx_x", "unsigned int"};
        Variable read_tile_y{"readTileIdx_y", "unsigned int"};
        auto     write_tile_x = read_tile_y;
        auto     write_tile_y = read_tile_x;

        return {Declaration{tile_size_x, 1},
                Declaration{tile_size_y, lengths[1] * lengths[2] / block_width},
                Declaration{tiles_per_batch, tile_size_x * tile_size_y},
                Declaration{read_tile_x, 0},
                Declaration{read_tile_y, Parens{tile % tiles_per_batch} / tile_size_x},
                AddAssign(offset_in, read_tile_x * stride_in[1]),
                AddAssign(offset_in, read_tile_y * block_width * stride_in[1]),
                AddAssign(offset_in, Parens{tile / tiles_per_batch} * stride_in[3]),
                AddAssign(offset_out, write_tile_x * block_width * stride_out[0]),
                AddAssign(offset_out, write_tile_y * stride_out[3]),
                AddAssign(offset_out, Parens{tile / tiles_per_batch} * stride_out[3]),
                Assign{offset_lds,
                       Parens{thread / threads_per_transform} * (length + lds_row_padding)}};
    }

    StatementList load_from_global_2d(bool load_registers)
    {
        // #-load for each thread to load all element in a tile
        auto num_load_blocks = (length * transforms_per_block) / threads_per_block;
        // #-row for a load block (global mem) = each thread will across these rows
        auto tid0_inc_step = transforms_per_block / num_load_blocks;

        StatementList stmts;
        stmts += Declaration{edge, "false"};
        stmts += Declaration{tid0};
        stmts += Declaration{tid1};

        stmts += If{transpose_type == "TILE_UNALIGNED",
                    {Assign{edge,
                            Ternary{Parens((tile_index + 1) * transforms_per_block > lengths[1]),
                                    "true",
                                    "false"}}}};

        // [dim0, dim1] = [tid0, tid1] :
        // each thread reads position [tid0, tid1], [tid0+step_h*1, tid1] , [tid0+step_h*2, tid1]...
        // tid0 walks the columns; tid1 walks the rows
        stmts += Assign{tid0, thread_id / length};
        stmts += Assign{tid1, thread_id % length};

        auto offset_tile_rbuf = [&](unsigned int i) {
            return tid1 * stride0 + (tid0 + i * tid0_inc_step) * stride_in[1];
        };
        auto offset_tile_wlds
            = [&](unsigned int i) { return tid1 * 1 + (tid0 + i * tid0_inc_step) * stride_lds; };

        StatementList regular_load;
        for(unsigned int i = 0; i < num_load_blocks; ++i)
            regular_load += Assign{lds_complex[offset_tile_wlds(i)],
                                   LoadGlobal{buf, offset_in + offset_tile_rbuf(i)}};

        StatementList edge_load;
        Variable      t{"t", "unsigned int"};
        edge_load += For{
            t,
            0,
            Parens{(tile_index * transforms_per_block + tid0 + t) < lengths[1]},
            tid0_inc_step,
            {Assign{lds_complex[tid1 * 1 + (tid0 + t) * stride_lds],
                    LoadGlobal{buf, offset_in + tid1 * stride0 + (tid0 + t) * stride_in[1]}}}};

        stmts += If{Or{transpose_type != "TILE_UNALIGNED", Not{edge}}, regular_load};
        stmts += Else{edge_load};

        return stmts;
    }

    StatementList store_to_global_2d(bool store_registers)
    {
        // #-column for a store block (global mem)
        auto store_block_w = transforms_per_block;
        // #-store for each thread to store all element in a tile
        auto num_store_blocks = (length * transforms_per_block) / threads_per_block;
        // #-row for a store block (global mem) = each thread will across these rows
        auto tid0_inc_step = length / num_store_blocks;

        StatementList stmts;

        // [dim0, dim1] = [tid0, tid1]:
        // each thread write GLOBAL_POS [tid0, tid1], [tid0+step_h*1, tid1] , [tid0+step_h*2, tid1].
        // NB: This is a transpose from LDS to global, so the pos of lds_read should be remapped
        stmts += Assign{tid0, thread_id / store_block_w};
        stmts += Assign{tid1, thread_id % store_block_w};

        auto offset_tile_wbuf = [&](unsigned int i) {
            return tid1 * stride_out[1] + (tid0 + i * tid0_inc_step) * stride0;
        };
        auto offset_tile_rlds
            = [&](unsigned int i) { return tid1 * stride_lds + (tid0 + i * tid0_inc_step) * 1; };

        StatementList regular_store;
        Expression    pred{tile_index * transforms_per_block + tid1 < lengths[1]};
        for(unsigned int i = 0; i < num_store_blocks; ++i)
            regular_store
                += StoreGlobal{buf, offset + offset_tile_wbuf(i), lds_complex[offset_tile_rlds(i)]};

        StatementList edge_store;
        edge_store += If{pred, regular_store};

        stmts += If{Or{transpose_type != "TILE_UNALIGNED", Not{edge}}, regular_store};
        stmts += Else{edge_store};

        return stmts;
    }

    StatementList load_from_global(bool load_registers) override
    {
        auto height = length * block_width / threads_per_block;

        StatementList stmts;
        stmts += Assign{thread, thread_id};

        // SBRC_3D_FFT_TRANS_Z_XY, SBRC_3D_FFT_ERC_TRANS_Z_XY
        StatementList load;
        for(unsigned int h = 0; h < height; ++h)
        {
            auto element = Parens{thread + h * threads_per_block};
            auto lidx    = element + Parens{Parens{element / length} * lds_row_padding};
            auto gidx    = offset_in + thread + h * threads_per_block;
            load += Assign{lds_complex[lidx], LoadGlobal{buf, gidx}};
        }
        stmts += If{
            Or{sbrc_type == "SBRC_3D_FFT_TRANS_Z_XY", sbrc_type == "SBRC_3D_FFT_ERC_TRANS_Z_XY"},
            load};

        // SBRC_3D_FFT_TRANS_XY_Z
        Variable tiles_per_batch{"tiles_per_batch", "unsigned int"};
        Variable tile_in_batch{"tile_in_batch", "unsigned int"};

        load.statements.clear();
        load += Declaration{tiles_per_batch,
                            lengths[1] * Parens{(lengths[2] + block_width - 1) / block_width}};
        load += Declaration{tile_in_batch, tile % tiles_per_batch};
        for(unsigned int h = 0; h < height; ++h)
        {
            auto lidx_constant = h % block_width * length;
            lidx_constant += (h / block_width) * threads_per_block;
            auto lidx = Literal{lidx_constant} + thread % length
                        + Parens{thread / length} * (height * length);
            auto gidx
                = offset_in + Parens{thread % length} * stride_in[0]
                  + (h / block_width * threads_per_block * stride_in[0])
                  + Parens{Parens{thread / length} * (threads_per_block / threads_per_transform)
                           + h % block_width}
                        * stride_in[2];
            auto idx = tile_in_batch / lengths[1] * block_width + h % block_width
                       + thread / length * block_width / threads_per_block;

            load += If{And{transpose_type == "TILE_UNALIGNED", idx >= lengths[2]},
                       StatementList{Break{}}};
            load += Assign{lds_complex[lidx], LoadGlobal{buf, gidx}};
        }
        load += Break{};
        stmts += If{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z", {While{"true", load}}};

        return stmts;
    }

    StatementList store_to_global(bool store_registers) override
    {
        StatementList stmts;

        auto threads_per_row = threads_per_block / block_width;
        auto height          = length * block_width / threads_per_block;

        // POSTPROCESSING SBRC_3D_FFT_ERC_TRANS_Z_XY
        StatementList post;
        Variable      null{"nullptr", "nullptr_t"};

        for(unsigned int h = 0; h < block_width; ++h)
        {
            post += Call{"post_process_interleaved_inplace",
                         {scalar_type, Variable{"true", ""}, Variable{"CallbackType::NONE", ""}},
                         {thread,
                          length - thread,
                          length,
                          length / 2,
                          lds_complex + (h * (length + lds_row_padding)),
                          0,
                          twiddles + length,
                          null,
                          null,
                          0,
                          null,
                          null}};
        }
        post += SyncThreads{};
        stmts += If{sbrc_type == "SBRC_3D_FFT_ERC_TRANS_Z_XY", post};

        StatementList store;
        // SBRC_3D_FFT_TRANS_XY_Z
        store.statements.clear();
        auto tiles_per_batch
            = Parens{lengths[1] * Parens{(lengths[2] + block_width - 1) / block_width}};
        auto tile_in_batch = tile % tiles_per_batch;

        for(unsigned int h = 0; h < height; ++h)
        {
            auto lidx = Literal{h * threads_per_row} + Parens{thread % block_width} * length
                        + Parens{thread / block_width};
            auto gidx = offset_out + Parens{thread % block_width} * stride_out[0]
                        + (Parens{thread / block_width} + h * threads_per_row) * stride_out[1];
            auto idx = tile_in_batch / lengths[1] * block_width + thread % block_width;
            store += If{Or{transpose_type != "TILE_UNALIGNED", idx < lengths[2]},
                        {StoreGlobal{buf, gidx, lds_complex[lidx]}}};
        }
        stmts += If{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z", store};

        // SBRC_3D_FFT_TRANS_Z_XY, SBRC_3D_FFT_ERC_TRANS_Z_XY
        store.statements.clear();
        for(unsigned int h = 0; h < height; ++h)
        {
            auto lidx = Literal{h * threads_per_row}
                        + Parens{thread % block_width} * (length + lds_row_padding)
                        + Parens{thread / block_width};
            auto gidx = offset_out + Parens{thread % block_width} * stride_out[0]
                        + (Parens{thread / block_width} + h * threads_per_row) * stride_out[2];
            store += StoreGlobal{buf, gidx, lds_complex[lidx]};
        }

        auto h    = height;
        auto lidx = Literal{h * threads_per_row}
                    + Parens{thread % block_width} * (length + lds_row_padding)
                    + Parens{thread / block_width};
        auto gidx = offset_out + Parens{thread % block_width} * stride_out[0]
                    + (Parens{thread / block_width} + h * threads_per_row) * stride_out[2];
        store += If{And{sbrc_type == "SBRC_3D_FFT_ERC_TRANS_Z_XY", thread < block_width},
                    {StoreGlobal{buf, gidx, lds_complex[lidx]}}};
        stmts += If{
            Or{sbrc_type == "SBRC_3D_FFT_TRANS_Z_XY", sbrc_type == "SBRC_3D_FFT_ERC_TRANS_Z_XY"},
            store};

        return stmts;
    }

    TemplateList device_templates() override
    {
        TemplateList tpls = StockhamKernel::device_templates();
        tpls.arguments.insert(tpls.arguments.begin() + 2, sbrc_type);
        tpls.arguments.insert(tpls.arguments.begin() + 3, transpose_type);
        return tpls;
    }

    TemplateList global_templates() override
    {
        TemplateList tpls = StockhamKernel::global_templates();
        tpls.arguments.insert(tpls.arguments.begin() + 2, sbrc_type);
        tpls.arguments.insert(tpls.arguments.begin() + 3, transpose_type);
        return tpls;
    }

    TemplateList device_call_templates() override
    {
        TemplateList tpls = StockhamKernel::device_call_templates();
        tpls.arguments.insert(tpls.arguments.begin() + 2, sbrc_type);
        tpls.arguments.insert(tpls.arguments.begin() + 3, transpose_type);
        return tpls;
    }

    std::vector<Expression> device_call_arguments(unsigned int call_iter) override
    {
        // TODO: We eventually will replace "length + lds_row_padding" with stride_lds
        //       after we finish refactoring all SBRC-type
        return {R,
                lds_real,
                lds_complex,
                twiddles,
                Literal{"1"},
                call_iter
                    ? Expression{offset_lds
                                 + call_iter * (length + lds_row_padding) * transforms_per_block}
                    : Expression{offset_lds},
                Literal{"true"}};
    }
};
