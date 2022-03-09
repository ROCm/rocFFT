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

    Variable stride0_out{"stride0_out", "const size_t"};

    //
    //
    //
    Variable len_along_block{"len_along_block", "const unsigned int"};
    Variable len_along_plane{"len_along_plane", "const unsigned int"};
    Variable stride_load_in{"stride_load_in", "const unsigned int"};
    Variable stride_store_out{"stride_store_out", "const unsigned int"};
    Variable stride_plane_in{"stride_plane_in", "const unsigned int"};
    Variable stride_plane_out{"stride_plane_out", "const unsigned int"};

    //
    // locals
    //
    Variable num_of_tiles_in_plane{"num_of_tiles_in_plane", "unsigned int"};
    Variable num_of_tiles_in_batch{"num_of_tiles_in_batch", "unsigned int"};
    Variable tile_index_in_plane{"tile_index_in_plane", "unsigned int"};

    Variable edge{"edge", "bool"};
    Variable tid1{"tid1", "unsigned int"};
    Variable tid0{"tid0", "unsigned int"};

    std::string tiling_name() override
    {
        return "SBRC";
    }

    StatementList check_batch() override
    {
        return {};
    }

    // we currently only use LDS padding for SBRC_3D_FFT_ERC_TRANS_Z_XY, so
    // there's no reason to look at the lds_padding parameter
    // otherwise.
    Expression get_lds_padding() override
    {
        return Ternary{sbrc_type != "SBRC_3D_FFT_ERC_TRANS_Z_XY", 0, lds_padding};
    }

    StatementList calculate_offsets() override
    {
        StatementList stmts;

        Variable plane_id{"plane_id", "unsigned int"};
        Variable tile_serial_in_batch{"tile_serial_in_batch", "unsigned int"};

        stmts += Declaration{
            len_along_block,
            Ternary{Parens{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z"}, lengths[2], lengths[1]}};
        stmts += Declaration{
            stride_load_in,
            Ternary{Parens{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z"}, stride_in[2], stride_in[1]}};
        stmts += Declaration{
            stride_store_out,
            Ternary{Parens{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z" || sbrc_type == "SBRC_2D"},
                    stride_out[1],
                    stride_out[2]}};

        stmts += LineBreak{};
        stmts += Declaration{num_of_tiles_in_batch};
        stmts += Declaration{tile_index_in_plane};

        stmts += LineBreak{};
        stmts
            += Declaration{num_of_tiles_in_plane, (len_along_block - 1) / transforms_per_block + 1};

        // --------------------------------------------------
        // SBRC_2D
        // --------------------------------------------------
        StatementList offset_2d;

        Variable d{"d", "unsigned int"};
        Variable index_along_d{"index_along_d", "unsigned int"};
        Variable remaining{"remaining", "unsigned int"};

        // offset_2d += CommentLines{"calculate offset for each tile:",
        //                       "  num_of_tiles_in_plane now means number of tiles along dim1",
        //                       "  tile_index_in_plane now means index of the tile along dim1"};
        // offset_2d += Assign{num_of_tiles_in_plane, (len_along_block - 1) / transforms_per_block + 1};

        offset_2d += Assign{num_of_tiles_in_batch, num_of_tiles_in_plane};
        offset_2d += Assign{tile_index_in_plane, block_id % num_of_tiles_in_plane};

        offset_2d += Declaration{remaining, block_id / num_of_tiles_in_plane};
        offset_2d += Declaration{index_along_d};

        offset_2d += Assign{offset_in, tile_index_in_plane * transforms_per_block * stride_load_in};
        offset_2d += Assign{offset_out, tile_index_in_plane * transforms_per_block * stride0_out};

        offset_2d += For{d,
                         2,
                         d < dim,
                         1,
                         {Assign{num_of_tiles_in_batch, num_of_tiles_in_batch * lengths[d]},
                          Assign{index_along_d, remaining % lengths[d]},
                          Assign{remaining, remaining / lengths[d]},
                          Assign{offset, offset + index_along_d * stride[d]}}};
        //
        // --------------------------------------------------
        // SBRC_3D
        // --------------------------------------------------
        StatementList offset_3d;

        // offset_xy_z
        //     += CommentLines{"calculate offset for each tile:",
        //                     " num_of_tiles_in_plane means number of tiles in a XZ-plane",
        //                     " num_of_tiles_in_batch means the total number of tiles in this batch",
        //                     " tile_index_in_plane means index of the tile in that XZ-plane",
        //                     " plane_id means the index of current XZ-plane (downwards along Y-axis)"};
        // offset_z_xy
        //     += CommentLines{"calculate offset for each tile:",
        //                     " num_of_tiles_in_plane means number of tiles in a xy plane",
        //                     " num_of_tiles_in_batch means the total number of tiles in this batch",
        //                     " tile_index_in_plane means index of the tile in that xy-plane",
        //                     " plane_id means the index of current xy-plane (inwards along Z-axis)"};

        offset_3d += Declaration{
            len_along_plane,
            Ternary{Parens{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z"}, lengths[1], lengths[2]}};
        offset_3d += Declaration{
            stride_plane_in,
            Ternary{Parens{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z"}, stride_in[1], stride_in[2]}};
        offset_3d += Declaration{
            stride_plane_out,
            Ternary{Parens{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z"}, stride_out[2], stride_out[1]}};
        stmts += Declaration{plane_id};
        stmts += Declaration{tile_serial_in_batch};

        // offset_3d += Assign{num_of_tiles_in_plane, (len_along_block - 1) / transforms_per_block + 1};
        offset_3d += Assign{num_of_tiles_in_batch, num_of_tiles_in_plane * len_along_plane};
        offset_3d += Assign{tile_serial_in_batch, block_id % num_of_tiles_in_batch};

        // Use DIAGONAL or NOT
        auto xy_z_diagonal = [&]() -> StatementList {
            return {
                Assign{tile_index_in_plane, tile_serial_in_batch % threads_per_transform},
                // looks like lengths[1] is correct? but if we restrict the size to be cubic, then we can simply use length
                Assign{plane_id,
                       (tile_serial_in_batch / threads_per_transform + tile_index_in_plane)
                           % length},
            };
        };
        auto xy_z_regular = [&]() -> StatementList {
            return {
                Assign{tile_index_in_plane, tile_serial_in_batch / lengths[1]},
                Assign{plane_id, block_id % lengths[1]},
                // Assign{tile_index_in_plane, block_id % num_of_tiles_in_plane},
                // Assign{plane_id, (block_id / num_of_tiles_in_plane) % lengths[1]},
            };
        };

        auto z_xy_diagonal = [&]() -> StatementList {
            return {
                Assign{tile_index_in_plane, tile_serial_in_batch % threads_per_transform},
                // looks like lengths[2] is correct? but if we restrict the size to be cubic, then we can simply use length
                Assign{plane_id,
                       (tile_serial_in_batch / threads_per_transform + tile_index_in_plane)
                           % length},
            };
        };
        auto z_xy_regular = [&]() -> StatementList {
            return {
                Assign{plane_id, tile_serial_in_batch / num_of_tiles_in_plane},
                Assign{tile_index_in_plane, block_id % num_of_tiles_in_plane},
            };
        };

        StatementList xy_z_offset;
        xy_z_offset += If{transpose_type == "DIAGONAL", xy_z_diagonal()};
        xy_z_offset += Else{xy_z_regular()};

        StatementList z_xy_offset;
        z_xy_offset += If{transpose_type == "DIAGONAL", z_xy_diagonal()};
        z_xy_offset += Else{z_xy_regular()};

        offset_3d += If{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z", xy_z_offset};
        offset_3d += Else{z_xy_offset};

        // Offset in/out/lds
        offset_3d += Assign{offset_in,
                            plane_id * stride_plane_in
                                + tile_index_in_plane * transforms_per_block * stride_load_in};
        offset_3d += Assign{offset_out,
                            plane_id * stride_plane_out
                                + tile_index_in_plane * transforms_per_block * stride0_out};

        stmts += If{sbrc_type == "SBRC_2D", offset_2d};
        stmts += Else{offset_3d};

        stmts += LineBreak{};
        stmts += Assign{transform,
                        tile_index_in_plane * transforms_per_block
                            + thread_id / threads_per_transform};
        stmts += Assign{batch, block_id / num_of_tiles_in_batch};
        stmts += Assign{offset, offset + batch * stride[dim]};
        stmts += Assign{stride_lds, (length + get_lds_padding())};
        stmts += Assign{offset_lds, stride_lds * (transform % transforms_per_block)};

        return stmts;
    }

    StatementList load_from_global(bool load_registers) override
    {
        StatementList stmts;

        stmts += Declaration{edge, "false"};
        stmts += Declaration{tid0};
        stmts += Declaration{tid1};

        // #-load for each thread to load all element in a tile
        auto num_load_blocks = (length * transforms_per_block) / workgroup_size;
        // #-row for a load block (global mem) = each thread will across these rows
        auto tid0_inc_step = transforms_per_block / num_load_blocks;
        // tpb/num_load_blocks, also = wgs/length, it's possible that they aren't divisible.
        bool divisible = (transforms_per_block % num_load_blocks) == 0;

        stmts += If{transpose_type == "TILE_UNALIGNED",
                    {Assign{edge,
                            Ternary{Parens((tile_index_in_plane + 1) * transforms_per_block
                                           > len_along_block),
                                    "true",
                                    "false"}}}};

        // [dim0, dim1] = [tid0, tid1] :
        // each thread reads position [tid0, tid1], [tid0+step_h*1, tid1] , [tid0+step_h*2, tid1]...
        // tid0 walks the columns; tid1 walks the rows
        if(divisible)
        {
            stmts += Assign{tid0, thread_id / length};
            stmts += Assign{tid1, thread_id % length};
        }

        // we need to take care about two diff cases for offset in buf and lds
        //  divisible: each load leads to a perfect block: update offset much simpler
        //  indivisible: need extra div and mod, otherwise each load will have some elements un-loaded:
        auto offset_tile_rbuf = [&](unsigned int i) {
            if(divisible)
                return tid1 * stride0 + (tid0 + i * tid0_inc_step) * stride_load_in;

            else
                return ((thread_id + i * workgroup_size) % length) * stride0
                       + ((thread_id + i * workgroup_size) / length) * stride_load_in;
        };
        auto offset_tile_wlds = [&](unsigned int i) {
            if(divisible)
                return tid1 * 1 + (tid0 + i * tid0_inc_step) * stride_lds;
            else
                return ((thread_id + i * workgroup_size) % length) * 1
                       + ((thread_id + i * workgroup_size) / length) * stride_lds;
        };

        StatementList regular_load;
        for(unsigned int i = 0; i < num_load_blocks; ++i)
            regular_load += Assign{lds_complex[offset_tile_wlds(i)],
                                   LoadGlobal{buf, offset_in + offset_tile_rbuf(i)}};

        StatementList edge_load;
        Variable      t{"t", "unsigned int"};
        if(divisible)
        {
            edge_load += For{
                t,
                0,
                Parens{(tile_index_in_plane * transforms_per_block + tid0 + t) < len_along_block},
                tid0_inc_step,
                {Assign{
                    lds_complex[tid1 * 1 + (tid0 + t) * stride_lds],
                    LoadGlobal{buf, offset_in + tid1 * stride0 + (tid0 + t) * stride_load_in}}}};
        }
        else
        {
            edge_load
                += For{t,
                       0,
                       Parens{(thread_id + t) < (length * transforms_per_block)},
                       workgroup_size,
                       {Assign{lds_complex[((thread_id + t) % length) * 1
                                           + ((thread_id + t) / length) * stride_lds],
                               LoadGlobal{buf,
                                          offset_in + ((thread_id + t) % length) * stride0
                                              + ((thread_id + t) / length) * stride_load_in}}}};
        }

        stmts += If{Or{transpose_type != "TILE_UNALIGNED", Not{edge}}, regular_load};
        stmts += Else{edge_load};

        return stmts;
    }

    StatementList store_to_global(bool store_registers) override
    {
        // #-column for a store block (global mem)
        auto store_block_w = transforms_per_block;
        // #-store for each thread to store all element in a tile
        auto num_store_blocks = (length * transforms_per_block) / workgroup_size;
        // #-row for a store block (global mem) = each thread will across these rows
        auto tid0_inc_step = length / num_store_blocks;
        // length / ((length * transforms_per_block) / workgroup_size) = wgs/tpb = blockwidth
        // so divisible should always true. But still put the logic here, since it's a
        // generator-wise if-else test, not inside the kernel code.
        bool divisible = (length % num_store_blocks) == 0;

        StatementList stmts;

        // POSTPROCESSING SBRC_3D_FFT_ERC_TRANS_Z_XY
        StatementList post;
        Variable      null{"nullptr", "nullptr_t"};
        for(unsigned int h = 0; h < transforms_per_block; ++h)
        {
            post += Call{"post_process_interleaved_inplace",
                         {scalar_type, Variable{"true", ""}, Variable{"CallbackType::NONE", ""}},
                         {thread_id,
                          length - thread_id,
                          length,
                          length / 2,
                          lds_complex + (h * stride_lds),
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

        // [dim0, dim1] = [tid0, tid1]:
        // each thread write GLOBAL_POS [tid0, tid1], [tid0+step_h*1, tid1] , [tid0+step_h*2, tid1].
        // NB: This is a transpose from LDS to global, so the pos of lds_read should be remapped
        if(divisible)
        {
            stmts += Assign{tid0, thread_id / store_block_w};
            stmts += Assign{tid1, thread_id % store_block_w};
        }

        // we need to take care about two diff cases for offset in buf and lds
        //  divisible: each store leads to a perfect block: update offset much simpler
        //  indivisible: need extra div and mod, otherwise each store will have some elements un-set:
        auto offset_tile_wbuf = [&](unsigned int i) {
            if(divisible)
                return tid1 * stride0 + (tid0 + i * tid0_inc_step) * stride_store_out;
            else
                return ((thread_id + i * workgroup_size) % store_block_w) * stride0
                       + (tid0 + i * tid0_inc_step) * stride_store_out;
        };
        auto offset_tile_rlds = [&](unsigned int i) {
            if(divisible)
                return tid1 * stride_lds + (tid0 + i * tid0_inc_step) * 1;
            else
                return ((thread_id + i * workgroup_size) % store_block_w) * stride_lds
                       + ((thread_id + i * workgroup_size) / store_block_w) * 1;
        };

        StatementList regular_store;
        Expression    pred{tile_index_in_plane * transforms_per_block + tid1 < len_along_block};
        for(unsigned int i = 0; i < num_store_blocks; ++i)
            regular_store
                += StoreGlobal{buf, offset + offset_tile_wbuf(i), lds_complex[offset_tile_rlds(i)]};

        // ERC_Z_XY
        auto i = num_store_blocks;
        regular_store += If{
            sbrc_type == "SBRC_3D_FFT_ERC_TRANS_Z_XY" && thread_id < transforms_per_block,
            {StoreGlobal{buf, offset + offset_tile_wbuf(i), lds_complex[offset_tile_rlds(i)]}}};

        StatementList edge_store;
        edge_store += If{pred, regular_store};

        stmts += If{Or{transpose_type != "TILE_UNALIGNED", Not{edge}}, regular_store};
        stmts += Else{edge_store};

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
};
