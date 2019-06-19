/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef POW2_LARGE_ENTRY_HIP_H
#define POW2_LARGE_ENTRY_HIP_H

#include "common.h"
#include "pow2_large.h"

//////////////////////////////////////////////////////////////////////////

// Local structure to embody/capture tile dimensions
typedef struct tag_Tile
{
    size_t x;
    size_t y;
} Tile;

template <typename T, int dir, int twl, TransTileDir ttd>
__global__ void transpose_var1(T*          twiddles_large,
                               T*          pmComplexIn,
                               T*          pmComplexOut,
                               const ulong numGroupsY,
                               const ulong stride_i,
                               const ulong stride_o,
                               const ulong dist_i,
                               const ulong dist_o)
{
    const Tile localIndex  = {(size_t)hipThreadIdx_x, (size_t)hipThreadIdx_y};
    const Tile localExtent = {(size_t)hipBlockDim_x, (size_t)hipBlockDim_y};
    const Tile groupIndex  = {(size_t)hipBlockIdx_x, (size_t)hipBlockIdx_y};

    // Calculate the unit address (in terms of datatype) of the beginning of the
    // Tile for the WG block
    // Transpose of input & output blocks happens with the Offset calculation
    const size_t reShapeFactor = 4;
    const size_t wgUnroll      = 16;
    const Tile   wgTileExtent  = {localExtent.x * reShapeFactor, localExtent.y / reShapeFactor};
    const size_t numGroupsY_1  = numGroupsY;
    // LDS is always complex and allocated transposed: lds[ wgTileExtent.y *
    // wgUnroll ][ wgTileExtent.x ];

    constexpr size_t TWIDTH = 64 / (sizeof(T) / 8);
    __shared__ T     lds[TWIDTH][TWIDTH];

    size_t currDimIndex;
    size_t rowSizeinUnits;

    size_t iOffset = 0;
    currDimIndex   = groupIndex.y;
    iOffset += (currDimIndex / numGroupsY_1) * dist_i;
    currDimIndex   = currDimIndex % numGroupsY_1;
    rowSizeinUnits = stride_i;

    if(ttd == TTD_IP_HOR)
    {
        iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;
        iOffset += groupIndex.x * wgTileExtent.x;
    }
    else
    {
        iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * groupIndex.x;
        iOffset += currDimIndex * wgTileExtent.x;
    }

    T* tileIn = pmComplexIn + iOffset;
    T  tmp;

    for(uint t = 0; t < wgUnroll; t++)
    {
        size_t xInd = localIndex.x + localExtent.x * (localIndex.y % wgTileExtent.y);
        size_t yInd = localIndex.y / wgTileExtent.y + t * wgTileExtent.y;
        size_t gInd = xInd + rowSizeinUnits * yInd;
        tmp         = tileIn[gInd];
        // Transpose of Tile data happens here

        if(ttd == TTD_IP_HOR)
        {
            if(twl == 3)
            {
                if(dir == -1)
                {
                    TWIDDLE_STEP_MUL_FWD(TWLstep3,
                                         twiddles_large,
                                         (groupIndex.x * wgTileExtent.x + xInd)
                                             * (currDimIndex * wgTileExtent.y * wgUnroll + yInd),
                                         tmp)
                }
                else
                {
                    TWIDDLE_STEP_MUL_INV(TWLstep3,
                                         twiddles_large,
                                         (groupIndex.x * wgTileExtent.x + xInd)
                                             * (currDimIndex * wgTileExtent.y * wgUnroll + yInd),
                                         tmp)
                }
            }
            else if(twl == 4)
            {
                if(dir == -1)
                {
                    TWIDDLE_STEP_MUL_FWD(TWLstep4,
                                         twiddles_large,
                                         (groupIndex.x * wgTileExtent.x + xInd)
                                             * (currDimIndex * wgTileExtent.y * wgUnroll + yInd),
                                         tmp)
                }
                else
                {
                    TWIDDLE_STEP_MUL_INV(TWLstep4,
                                         twiddles_large,
                                         (groupIndex.x * wgTileExtent.x + xInd)
                                             * (currDimIndex * wgTileExtent.y * wgUnroll + yInd),
                                         tmp)
                }
            }
        }
        else
        {
            if(twl == 3)
            {
                if(dir == -1)
                {
                    TWIDDLE_STEP_MUL_FWD(TWLstep3,
                                         twiddles_large,
                                         (groupIndex.x * wgTileExtent.y * wgUnroll + yInd)
                                             * (currDimIndex * wgTileExtent.x + xInd),
                                         tmp)
                }
                else
                {
                    TWIDDLE_STEP_MUL_INV(TWLstep3,
                                         twiddles_large,
                                         (groupIndex.x * wgTileExtent.y * wgUnroll + yInd)
                                             * (currDimIndex * wgTileExtent.x + xInd),
                                         tmp)
                }
            }
            else if(twl == 4)
            {
                if(dir == -1)
                {
                    TWIDDLE_STEP_MUL_FWD(TWLstep4,
                                         twiddles_large,
                                         (groupIndex.x * wgTileExtent.y * wgUnroll + yInd)
                                             * (currDimIndex * wgTileExtent.x + xInd),
                                         tmp)
                }
                else
                {
                    TWIDDLE_STEP_MUL_INV(TWLstep4,
                                         twiddles_large,
                                         (groupIndex.x * wgTileExtent.y * wgUnroll + yInd)
                                             * (currDimIndex * wgTileExtent.x + xInd),
                                         tmp)
                }
            }
        }

        lds[xInd][yInd] = tmp;
    }

    __syncthreads();

    size_t oOffset = 0;
    currDimIndex   = groupIndex.y;
    oOffset += (currDimIndex / numGroupsY_1) * dist_o;
    currDimIndex   = currDimIndex % numGroupsY_1;
    rowSizeinUnits = stride_o;

    if(ttd == TTD_IP_HOR)
    {
        oOffset += rowSizeinUnits * wgTileExtent.x * groupIndex.x;
        oOffset += currDimIndex * wgTileExtent.y * wgUnroll;
    }
    else
    {
        oOffset += rowSizeinUnits * wgTileExtent.x * currDimIndex;
        oOffset += groupIndex.x * wgTileExtent.y * wgUnroll;
    }

    T* tileOut = pmComplexOut + oOffset;

    const size_t transposeRatio = wgTileExtent.x / (wgTileExtent.y * wgUnroll);
    const size_t groupingPerY   = wgUnroll / wgTileExtent.y;

    for(uint t = 0; t < wgUnroll; t++)
    {
        size_t xInd   = localIndex.x + localExtent.x * (localIndex.y % groupingPerY);
        size_t yInd   = localIndex.y / groupingPerY + t * (wgTileExtent.y * transposeRatio);
        tmp           = lds[yInd][xInd];
        size_t gInd   = xInd + rowSizeinUnits * yInd;
        tileOut[gInd] = tmp;
    }
}

#endif // POW2_LARGE_ENTRY_HIP_H
