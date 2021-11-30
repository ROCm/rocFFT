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
        n_device_calls = block_width / (threads_per_block / threads_per_transform);
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

    std::string tiling_name() override
    {
        return "SBRC";
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

        stmts += If{sbrc_type == "SBRC_2D", calculate_offsets_2d()};
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
        Variable current_length{"current_length", "unsigned int"};
        Variable remaining{"remaining", "unsigned int"};
        Variable i{"i", "unsigned int"};
        Variable j{"j", "unsigned int"};

        StatementList offsets;
        offsets += Declaration{current_length};
        offsets += Declaration{remaining};
        offsets += Assign{remaining, tile};

        offsets += For{i,
                       dim,
                       i > 2,
                       -1,
                       {Assign{current_length, 1},
                        For{j, 2, j < i, 1, {MultiplyAssign(current_length, lengths[j])}},
                        MultiplyAssign(current_length, lengths[1] / block_width),
                        AddAssign(offset_in, Parens{remaining / current_length} * stride_in[i]),
                        AddAssign(offset_out, Parens{remaining / current_length} * stride_out[i]),
                        Assign{remaining, remaining % current_length}}};

        offsets += Assign{current_length, lengths[1] / block_width};
        offsets += AddAssign(offset_in, Parens{remaining / current_length} * stride_in[2]);
        offsets += AddAssign(offset_in,
                             Parens{remaining % current_length} * Parens{block_width * lengths[0]});

        offsets += AddAssign(offset_out, Parens{remaining / current_length} * stride_out[2]);
        offsets += AddAssign(
            offset_out, Parens{remaining % current_length} * Parens{block_width * stride_out[1]});

        offsets += Assign{offset_lds,
                          Parens{thread / threads_per_transform} * (length + lds_row_padding)};
        return offsets;
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

    StatementList load_from_global(bool load_registers) override
    {
        auto height = length * block_width / threads_per_block;

        StatementList stmts;
        stmts += Assign{thread, thread_id};

        // SBRC_2D, SBRC_3D_FFT_TRANS_Z_XY, SBRC_3D_FFT_ERC_TRANS_Z_XY
        StatementList load;
        for(unsigned int h = 0; h < height; ++h)
        {
            auto element = Parens{thread + h * threads_per_block};
            auto lidx    = element + Parens{Parens{element / length} * lds_row_padding};
            auto gidx    = offset_in + thread + h * threads_per_block;
            load += Assign{lds_complex[lidx], LoadGlobal{buf, gidx}};
        }
        stmts += If{Or{sbrc_type == "SBRC_2D",
                       sbrc_type == "SBRC_3D_FFT_TRANS_Z_XY",
                       sbrc_type == "SBRC_3D_FFT_ERC_TRANS_Z_XY"},
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

        // SBRC_2D
        StatementList store;
        for(unsigned int h = 0; h < height; ++h)
        {
            auto row  = Parens{(thread + h * threads_per_block) / block_width};
            auto col  = Parens{thread % block_width};
            auto lidx = col * length + row;
            auto gidx = offset_out + row * stride_out[0] + col * stride_out[1];
            store += StoreGlobal{buf, gidx, lds_complex[lidx]};
        }
        stmts += If{sbrc_type == "SBRC_2D", store};

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
                stride_lds,
                call_iter
                    ? Expression{offset_lds
                                 + call_iter * (length + lds_row_padding) * transforms_per_block}
                    : Expression{offset_lds},
                Literal{"true"}};
    }
};
