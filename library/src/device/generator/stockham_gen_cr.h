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

struct StockhamKernelCR : public StockhamKernel
{
    explicit StockhamKernelCR(StockhamGeneratorSpecs& specs)
        : StockhamKernel(specs)
    {
    }

    //
    // locals
    //
    Variable tile_index{"tile_index", "size_t"};
    Variable tile_length{"tile_length", "size_t"};
    Variable edge{"edge", "bool"};
    Variable tid1{"tid1", "size_t"};
    Variable tid0{"tid0", "size_t"};

    std::string tiling_name() override
    {
        return "SBCR";
    }

    StatementList calculate_offsets() override
    {
        Variable d{"d", "int"};
        Variable index_along_d{"index_along_d", "size_t"};
        Variable remaining{"remaining", "size_t"};
        Variable plength{"plength", "size_t"};

        StatementList stmts;
        stmts += Declaration{tile_index};
        stmts += Declaration{tile_length};
        stmts += LineBreak{};
        stmts += CommentLines{"calculate offset for each tile:",
                              "  tile_index  now means index of the tile along dim1",
                              "  tile_length now means number of tiles along dim1"};

        stmts += Declaration{plength, 1};
        stmts += Declaration{remaining};
        stmts += Declaration{index_along_d};

        stmts += Assign{tile_length, (lengths[1] - 1) / transforms_per_block + 1};
        stmts += Assign{plength, tile_length};
        stmts += Assign{tile_index, block_id % tile_length};
        stmts += Assign{remaining, block_id / tile_length};
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
        stmts += Assign{stride_lds, (length + get_lds_padding())};
        stmts += Assign{offset_lds, stride_lds * (transform % transforms_per_block)};

        return stmts;
    }

    StatementList load_from_global(bool load_registers) override
    {
        auto stripmine_w = transforms_per_block;
        auto stripmine_h = workgroup_size / stripmine_w;

        StatementList stmts;
        stmts += Declaration{edge};
        stmts += Declaration{tid0};
        stmts += Declaration{tid1};
        stmts += Assign{
            edge,
            Ternary{Parens{Greater{Parens{tile_index + 1} * transforms_per_block, lengths[1]}},
                    "true",
                    "false"}};
        // tid0 walks the columns; tid1 walks the rows
        stmts += Assign{tid1, thread_id % stripmine_w};
        stmts += Assign{tid0, thread_id / stripmine_w};

        auto offset_tile_rbuf
            = [&](unsigned int i) { return tid1 * stride[1] + (tid0 + i * stripmine_h) * stride0; };
        auto offset_tile_wlds
            = [&](unsigned int i) { return tid1 * stride_lds + (tid0 + i * stripmine_h) * 1; };

        StatementList regular_load;

        for(unsigned int i = 0; i < length / stripmine_h; ++i)
            regular_load += Assign{lds_complex[offset_tile_wlds(i)],
                                   LoadGlobal{buf, offset + offset_tile_rbuf(i)}};

        StatementList stmts_c2real_pre_no_edge;
        stmts_c2real_pre_no_edge += regular_load;
        stmts_c2real_pre_no_edge += LineBreak{};
        stmts_c2real_pre_no_edge += CommentLines{
            "append extra global loading for C2Real pre-process only, one more element per col."};

        stmts_c2real_pre_no_edge
            += If{Equal{embedded_type, "EmbeddedType::C2Real_PRE"},
                  {If{Less{thread_id, transforms_per_block},
                      {Assign{lds_complex[tid1 * stride_lds + length],
                              LoadGlobal{buf, offset + offset_tile_rbuf(length / stripmine_h)}}}}}};

        stmts += If{Not{edge}, stmts_c2real_pre_no_edge};

        StatementList stmts_c2real_pre_edge;
        stmts_c2real_pre_edge += regular_load;
        stmts_c2real_pre_edge += LineBreak{};
        stmts_c2real_pre_edge += CommentLines{
            "append extra global loading for C2Real pre-process only, one more element per col."};

        stmts_c2real_pre_edge
            += If{Equal{embedded_type, "EmbeddedType::C2Real_PRE"},
                  {If{Less{thread_id,
                           Parens{transforms_per_block
                                  - (tile_index + 1) * transforms_per_block % lengths[1]}},
                      {Assign{lds_complex[tid1 * stride_lds + length],
                              LoadGlobal{buf, offset + offset_tile_rbuf(length / stripmine_h)}}}}}};

        stmts += If{
            edge,
            {If{tile_index * transforms_per_block + tid1 < lengths[1], stmts_c2real_pre_edge}}};

        return stmts;
    }

    StatementList store_to_global(bool store_registers) override
    {
        StatementList stmts;

        StatementList regular_store;
        for(unsigned int i = 0; i < length / threads_per_transform; ++i)
        {
            regular_store += Assign{tid0, (i * workgroup_size + thread_id) % length};
            regular_store += Assign{tid1, (i * workgroup_size + thread_id) / length};
            regular_store += StoreGlobal{buf,
                                         offset + tid1 * stride[1] + tid0 * stride0,
                                         lds_complex[tid1 * stride_lds + tid0]};
        }
        stmts += If{Not{edge}, regular_store};

        StatementList partial_store;
        Variable      t{"t", "int"};
        partial_store += For{t,
                             0,
                             Parens{(t * workgroup_size + thread_id) / length}
                                 < Parens{transforms_per_block
                                          - (tile_index + 1) * transforms_per_block % lengths[1]},
                             1,
                             {Assign{tid0, (t * workgroup_size + thread_id) % length},
                              Assign{tid1, (t * workgroup_size + thread_id) / length},
                              StoreGlobal{buf,
                                          offset + tid1 * stride[1] + tid0 * stride0,
                                          lds_complex[tid1 * stride_lds + tid0]}}};
        stmts += If{edge, partial_store};

        return stmts;
    }
};
