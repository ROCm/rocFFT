#include "../../../shared/array_predicate.h"
#include "../include/arithmetic.h"
#include "array_format.h"
#include "kernel_launch.h"
#include "rocfft_hip.h"
#include <numeric>

// template flag to say whether the kernel only needs to consider two
// dimensions, or 3 (or more).  helps avoid looping logic in the
// common case.
enum TransposeDim
{
    TransposeDim2,
    TransposeDim3,
    TransposeDim4Plus,
};

// chain of macros to iterate over transpose kernel template parameters, to
// set a function pointer 'kernel_func'
#define TRANSPOSE_KERNEL_CBTYPE(T_I, T_O, TILE_X, TILE_Y, DIM, TWL, DIR, DIAG, ALIGN, CBTYPE) \
    kernel_func = transpose_kernel<TILE_X, TILE_Y, T_I, T_O, DIM, TWL, DIR, DIAG, ALIGN, CBTYPE>;

#define TRANSPOSE_KERNEL_DIR(T_I, T_O, TILE_X, TILE_Y, DIM, TWL, DIR, DIAG, ALIGN)        \
    {                                                                                     \
        if(cbtype == CallbackType::NONE)                                                  \
            TRANSPOSE_KERNEL_CBTYPE(                                                      \
                T_I, T_O, TILE_X, TILE_Y, DIM, TWL, DIR, DIAG, ALIGN, CallbackType::NONE) \
        else                                                                              \
            TRANSPOSE_KERNEL_CBTYPE(T_I,                                                  \
                                    T_O,                                                  \
                                    TILE_X,                                               \
                                    TILE_Y,                                               \
                                    DIM,                                                  \
                                    TWL,                                                  \
                                    DIR,                                                  \
                                    DIAG,                                                 \
                                    ALIGN,                                                \
                                    CallbackType::USER_LOAD_STORE)                        \
    }

#define TRANSPOSE_KERNEL_TWL(T_I, T_O, TILE_X, TILE_Y, DIM, TWL, DIAG, ALIGN)         \
    {                                                                                 \
        if(dir == 1)                                                                  \
            TRANSPOSE_KERNEL_DIR(T_I, T_O, TILE_X, TILE_Y, DIM, TWL, 1, DIAG, ALIGN)  \
        else                                                                          \
            TRANSPOSE_KERNEL_DIR(T_I, T_O, TILE_X, TILE_Y, DIM, TWL, -1, DIAG, ALIGN) \
    }

// 2D transpose needs large twiddle, 3D transposes don't need it.  2D
// transpose kernels normally only run with dim == 2, but in theory
// they could be decomposed into higher dimension kernels too.
#define TRANSPOSE_KERNEL_DIM(T_I, T_O, TILE_X, TILE_Y, DIAG, ALIGN)                               \
    {                                                                                             \
        if(data->node->scheme == CS_KERNEL_TRANSPOSE)                                             \
        {                                                                                         \
            if(twl == 0)                                                                          \
            {                                                                                     \
                if(length.size() == 2)                                                            \
                    TRANSPOSE_KERNEL_DIR(                                                         \
                        T_I, T_O, TILE_X, TILE_Y, TransposeDim2, 0, -1, DIAG, ALIGN)              \
                else if(length.size() == 3)                                                       \
                    TRANSPOSE_KERNEL_DIR(                                                         \
                        T_I, T_O, TILE_X, TILE_Y, TransposeDim3, 0, -1, DIAG, ALIGN)              \
                else                                                                              \
                    TRANSPOSE_KERNEL_DIR(                                                         \
                        T_I, T_O, TILE_X, TILE_Y, TransposeDim4Plus, 0, -1, DIAG, ALIGN)          \
            }                                                                                     \
            if(twl == 1)                                                                          \
            {                                                                                     \
                if(length.size() == 2)                                                            \
                    TRANSPOSE_KERNEL_TWL(T_I, T_O, TILE_X, TILE_Y, TransposeDim2, 1, DIAG, ALIGN) \
                else if(length.size() == 3)                                                       \
                    TRANSPOSE_KERNEL_TWL(T_I, T_O, TILE_X, TILE_Y, TransposeDim3, 1, DIAG, ALIGN) \
                else                                                                              \
                    TRANSPOSE_KERNEL_TWL(                                                         \
                        T_I, T_O, TILE_X, TILE_Y, TransposeDim4Plus, 1, DIAG, ALIGN)              \
            }                                                                                     \
            if(twl == 2)                                                                          \
            {                                                                                     \
                if(length.size() == 2)                                                            \
                    TRANSPOSE_KERNEL_TWL(T_I, T_O, TILE_X, TILE_Y, TransposeDim2, 2, DIAG, ALIGN) \
                else if(length.size() == 3)                                                       \
                    TRANSPOSE_KERNEL_TWL(T_I, T_O, TILE_X, TILE_Y, TransposeDim3, 2, DIAG, ALIGN) \
                else                                                                              \
                    TRANSPOSE_KERNEL_TWL(                                                         \
                        T_I, T_O, TILE_X, TILE_Y, TransposeDim4Plus, 2, DIAG, ALIGN)              \
            }                                                                                     \
            if(twl == 3)                                                                          \
            {                                                                                     \
                if(length.size() == 2)                                                            \
                    TRANSPOSE_KERNEL_TWL(T_I, T_O, TILE_X, TILE_Y, TransposeDim2, 3, DIAG, ALIGN) \
                else if(length.size() == 3)                                                       \
                    TRANSPOSE_KERNEL_TWL(T_I, T_O, TILE_X, TILE_Y, TransposeDim3, 3, DIAG, ALIGN) \
                else                                                                              \
                    TRANSPOSE_KERNEL_TWL(                                                         \
                        T_I, T_O, TILE_X, TILE_Y, TransposeDim4Plus, 3, DIAG, ALIGN)              \
            }                                                                                     \
            if(twl == 4)                                                                          \
            {                                                                                     \
                if(length.size() == 2)                                                            \
                    TRANSPOSE_KERNEL_TWL(T_I, T_O, TILE_X, TILE_Y, TransposeDim2, 4, DIAG, ALIGN) \
                else if(length.size() == 3)                                                       \
                    TRANSPOSE_KERNEL_TWL(T_I, T_O, TILE_X, TILE_Y, TransposeDim3, 4, DIAG, ALIGN) \
                else                                                                              \
                    TRANSPOSE_KERNEL_TWL(                                                         \
                        T_I, T_O, TILE_X, TILE_Y, TransposeDim4Plus, 4, DIAG, ALIGN)              \
            }                                                                                     \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            if(cbtype == CallbackType::NONE)                                                      \
            {                                                                                     \
                if(length.size() == 3)                                                            \
                    TRANSPOSE_KERNEL_CBTYPE(T_I,                                                  \
                                            T_O,                                                  \
                                            TILE_X,                                               \
                                            TILE_Y,                                               \
                                            TransposeDim3,                                        \
                                            0,                                                    \
                                            -1,                                                   \
                                            DIAG,                                                 \
                                            ALIGN,                                                \
                                            CallbackType::NONE)                                   \
                else                                                                              \
                    TRANSPOSE_KERNEL_CBTYPE(T_I,                                                  \
                                            T_O,                                                  \
                                            TILE_X,                                               \
                                            TILE_Y,                                               \
                                            TransposeDim4Plus,                                    \
                                            0,                                                    \
                                            -1,                                                   \
                                            DIAG,                                                 \
                                            ALIGN,                                                \
                                            CallbackType::NONE)                                   \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                if(length.size() == 3)                                                            \
                    TRANSPOSE_KERNEL_CBTYPE(T_I,                                                  \
                                            T_O,                                                  \
                                            TILE_X,                                               \
                                            TILE_Y,                                               \
                                            TransposeDim3,                                        \
                                            0,                                                    \
                                            -1,                                                   \
                                            DIAG,                                                 \
                                            ALIGN,                                                \
                                            CallbackType::USER_LOAD_STORE)                        \
                else                                                                              \
                    TRANSPOSE_KERNEL_CBTYPE(T_I,                                                  \
                                            T_O,                                                  \
                                            TILE_X,                                               \
                                            TILE_Y,                                               \
                                            TransposeDim4Plus,                                    \
                                            0,                                                    \
                                            -1,                                                   \
                                            DIAG,                                                 \
                                            ALIGN,                                                \
                                            CallbackType::USER_LOAD_STORE)                        \
            }                                                                                     \
        }                                                                                         \
    }

// decide if the transpose lengths are tile-aligned
#define TRANSPOSE_KERNEL_ALIGN(T_I, T_O, TILE_X, TILE_Y, DIAG)          \
    {                                                                   \
        if(length[0] % TILE_X == 0 && length[1] % TILE_X == 0)          \
            TRANSPOSE_KERNEL_DIM(T_I, T_O, TILE_X, TILE_Y, DIAG, true)  \
        else                                                            \
            TRANSPOSE_KERNEL_DIM(T_I, T_O, TILE_X, TILE_Y, DIAG, false) \
    }

#define TRANSPOSE_KERNEL_DIAG(T_I, T_O, TILE_X, TILE_Y)             \
    {                                                               \
        if(diagonal)                                                \
            TRANSPOSE_KERNEL_ALIGN(T_I, T_O, TILE_X, TILE_Y, true)  \
        else                                                        \
            TRANSPOSE_KERNEL_ALIGN(T_I, T_O, TILE_X, TILE_Y, false) \
    }

// single precision uses 64x64 tile, 64x16 threads
static const unsigned int TILE_X_SINGLE = 64;
static const unsigned int TILE_Y_SINGLE = 16;

// double precision uses 32x32 tile, 32x32 threads
static const unsigned int TILE_X_DOUBLE = 32;
static const unsigned int TILE_Y_DOUBLE = 32;

// declare a function pointer for the specified precision, then
// invoke macros above to assign it and launch the kernel
#define LAUNCH_TRANSPOSE_KERNEL(T_I, T_O)                                               \
    if(data->node->precision == rocfft_precision_single)                                \
    {                                                                                   \
        decltype(&transpose_kernel<TILE_X_SINGLE,                                       \
                                   TILE_Y_SINGLE,                                       \
                                   T_I<float2>,                                         \
                                   T_O<float2>,                                         \
                                   TransposeDim2,                                       \
                                   4,                                                   \
                                   -1,                                                  \
                                   false,                                               \
                                   false,                                               \
                                   CallbackType::NONE>) kernel_func                     \
            = nullptr;                                                                  \
                                                                                        \
        grid    = {DivRoundingUp<unsigned int>(length[0], TILE_X_SINGLE),               \
                DivRoundingUp<unsigned int>(gridYrows, TILE_X_SINGLE),               \
                gridZ};                                                              \
        threads = {TILE_X_SINGLE, TILE_Y_SINGLE};                                       \
        TRANSPOSE_KERNEL_DIAG(T_I<float2>, T_O<float2>, TILE_X_SINGLE, TILE_Y_SINGLE)   \
                                                                                        \
        hipLaunchKernelGGL(kernel_func,                                                 \
                           grid,                                                        \
                           threads,                                                     \
                           0,                                                           \
                           data->rocfft_stream,                                         \
                           {data->bufIn[0], data->bufIn[1]},                            \
                           {data->bufOut[0], data->bufOut[1]},                          \
                           static_cast<const float2*>(data->node->twiddles_large),      \
                           length.size(),                                               \
                           length[0],                                                   \
                           length[1],                                                   \
                           length.size() > 2 ? length[2] : 1,                           \
                           kargs_lengths(data->node->devKernArg),                       \
                           istride[0],                                                  \
                           istride[1],                                                  \
                           istride.size() > 2 ? istride[2] : 0,                         \
                           kargs_stride_in(data->node->devKernArg),                     \
                           data->node->iDist,                                           \
                           ostride[0],                                                  \
                           ostride[1],                                                  \
                           ostride.size() > 2 ? ostride[2] : 0,                         \
                           kargs_stride_out(data->node->devKernArg),                    \
                           data->node->oDist,                                           \
                           data->callbacks.load_cb_fn,                                  \
                           data->callbacks.load_cb_data,                                \
                           data->callbacks.load_cb_lds_bytes,                           \
                           data->callbacks.store_cb_fn,                                 \
                           data->callbacks.store_cb_data);                              \
    }                                                                                   \
    else                                                                                \
    {                                                                                   \
        decltype(&transpose_kernel<TILE_X_DOUBLE,                                       \
                                   TILE_Y_DOUBLE,                                       \
                                   T_I<double2>,                                        \
                                   T_O<double2>,                                        \
                                   TransposeDim2,                                       \
                                   4,                                                   \
                                   -1,                                                  \
                                   false,                                               \
                                   false,                                               \
                                   CallbackType::NONE>) kernel_func                     \
            = nullptr;                                                                  \
                                                                                        \
        grid    = {DivRoundingUp<unsigned int>(length[0], TILE_X_DOUBLE),               \
                DivRoundingUp<unsigned int>(gridYrows, TILE_X_DOUBLE),               \
                gridZ};                                                              \
        threads = {TILE_X_DOUBLE, TILE_Y_DOUBLE};                                       \
        TRANSPOSE_KERNEL_DIAG(T_I<double2>, T_O<double2>, TILE_X_DOUBLE, TILE_Y_DOUBLE) \
                                                                                        \
        hipLaunchKernelGGL(kernel_func,                                                 \
                           grid,                                                        \
                           threads,                                                     \
                           0,                                                           \
                           data->rocfft_stream,                                         \
                           {data->bufIn[0], data->bufIn[1]},                            \
                           {data->bufOut[0], data->bufOut[1]},                          \
                           static_cast<const double2*>(data->node->twiddles_large),     \
                           length.size(),                                               \
                           length[0],                                                   \
                           length[1],                                                   \
                           length.size() > 2 ? length[2] : 1,                           \
                           kargs_lengths(data->node->devKernArg),                       \
                           istride[0],                                                  \
                           istride[1],                                                  \
                           istride.size() > 2 ? istride[2] : 0,                         \
                           kargs_stride_in(data->node->devKernArg),                     \
                           data->node->iDist,                                           \
                           ostride[0],                                                  \
                           ostride[1],                                                  \
                           ostride.size() > 2 ? ostride[2] : 0,                         \
                           kargs_stride_out(data->node->devKernArg),                    \
                           data->node->oDist,                                           \
                           data->callbacks.load_cb_fn,                                  \
                           data->callbacks.load_cb_data,                                \
                           data->callbacks.load_cb_lds_bytes,                           \
                           data->callbacks.store_cb_fn,                                 \
                           data->callbacks.store_cb_data);                              \
    }

template <unsigned int TILE_X,
          unsigned int TILE_Y,
          typename T_I,
          typename T_O,
          TransposeDim DIM,
          int          TWL,
          int          TWL_DIR,
          bool         DIAGONAL,
          bool         TILE_ALIGNED,
          CallbackType cbtype>
__global__ __launch_bounds__(TILE_X* TILE_Y) void transpose_kernel(
    const T_I input,
    const T_O output,
    const typename T_I::complex_type* __restrict__ twiddles_large,
    unsigned int dim,
    unsigned int length0,
    unsigned int length1,
    unsigned int length2,
    const size_t* __restrict__ lengths,
    unsigned int stride_in0,
    unsigned int stride_in1,
    unsigned int stride_in2,
    const size_t* __restrict__ stride_in,
    unsigned int idist,
    unsigned int stride_out0,
    unsigned int stride_out1,
    unsigned int stride_out2,
    const size_t* __restrict__ stride_out,
    unsigned int odist,
    void* __restrict__ load_cb_fn,
    void* __restrict__ load_cb_data,
    uint32_t load_cb_lds_bytes,
    void* __restrict__ store_cb_fn,
    void* __restrict__ store_cb_data)
{
    typedef typename T_I::complex_type T;

    // we use TILE_X*TILE_X tiles - TILE_Y must evenly divide into
    // TILE_X, so that ELEMS_PER_THREAD is integral
    static_assert(TILE_X % TILE_Y == 0);
    static const auto ELEMS_PER_THREAD = TILE_X / TILE_Y;
    static_assert(ELEMS_PER_THREAD > 0);
    __shared__ T lds[TILE_X][TILE_X];

    unsigned int tileBlockIdx_y = blockIdx.y;
    unsigned int tileBlockIdx_x = blockIdx.x;

    if(DIAGONAL)
    {
        auto bid       = blockIdx.x + gridDim.x * blockIdx.y;
        tileBlockIdx_y = bid % gridDim.y;
        tileBlockIdx_x = (bid / gridDim.y + tileBlockIdx_y) % gridDim.x;
    }

    // if only using 2 dimensions, pretend length2 is 1 so the
    // compiler can optimize out comparisons against it
    if(DIM == TransposeDim2)
        length2 = 1;

    unsigned int tile_x_index = threadIdx.x;
    unsigned int tile_y_index = threadIdx.y;

    // work out offset for dimensions after the first 3
    unsigned int remaining  = blockIdx.z;
    unsigned int offset_in  = 0;
    unsigned int offset_out = 0;

    // use template-specified dim to avoid loops if possible
    if(DIM == TransposeDim4Plus)
    {
        for(int d = 3; d < dim; ++d)
        {
            auto index_along_d = remaining % lengths[d];
            remaining          = remaining / lengths[d];
            offset_in          = offset_in + index_along_d * stride_in[d];
            offset_out         = offset_out + index_along_d * stride_out[d];
        }
    }

    // remaining is now the batch
    offset_in += remaining * idist;
    offset_out += remaining * odist;

#pragma unroll
    for(unsigned int i = 0; i < ELEMS_PER_THREAD; ++i)
    {
        auto logical_row = TILE_X * tileBlockIdx_y + tile_y_index + i * TILE_Y;

        auto idx0 = TILE_X * tileBlockIdx_x + tile_x_index;
        auto idx1 = logical_row;
        if(DIM != TransposeDim2)
            idx1 %= length1;
        auto idx2 = DIM == TransposeDim2 ? 0 : logical_row / length1;

        if(!TILE_ALIGNED && (idx0 >= length0 || idx1 >= length1 || idx2 >= length2))
            break;

        auto global_read_idx
            = idx0 * stride_in0 + idx1 * stride_in1 + idx2 * stride_in2 + offset_in;
        auto elem = Handler<T_I, cbtype>::read(input, global_read_idx, load_cb_fn, load_cb_data);

        auto twl_idx = idx0 * idx1;
        if(TWL == 1)
        {
            if(TWL_DIR == -1)
            {
                TWIDDLE_STEP_MUL_FWD(TWLstep1, twiddles_large, twl_idx, elem);
            }
            else
            {
                TWIDDLE_STEP_MUL_INV(TWLstep1, twiddles_large, twl_idx, elem);
            }
        }
        else if(TWL == 2)
        {
            if(TWL_DIR == -1)
            {
                TWIDDLE_STEP_MUL_FWD(TWLstep2, twiddles_large, twl_idx, elem);
            }
            else
            {
                TWIDDLE_STEP_MUL_INV(TWLstep2, twiddles_large, twl_idx, elem);
            }
        }
        else if(TWL == 3)
        {
            if(TWL_DIR == -1)
            {
                TWIDDLE_STEP_MUL_FWD(TWLstep3, twiddles_large, twl_idx, elem);
            }
            else
            {
                TWIDDLE_STEP_MUL_INV(TWLstep3, twiddles_large, twl_idx, elem);
            }
        }
        else if(TWL == 4)
        {
            if(TWL_DIR == -1)
            {
                TWIDDLE_STEP_MUL_FWD(TWLstep4, twiddles_large, twl_idx, elem);
            }
            else
            {
                TWIDDLE_STEP_MUL_INV(TWLstep4, twiddles_large, twl_idx, elem);
            }
        }
        lds[tile_x_index][i * TILE_Y + tile_y_index] = elem;
    }

    __syncthreads();
    T val[ELEMS_PER_THREAD];

    // reallocate threads to write along fastest dim (length1) and

    // read transposed from LDS
    tile_x_index = threadIdx.y;
    tile_y_index = threadIdx.x;

#pragma unroll
    for(unsigned int i = 0; i < ELEMS_PER_THREAD; ++i)
    {
        val[i] = lds[tile_x_index + i * TILE_Y][tile_y_index];
    }

#pragma unroll
    for(unsigned int i = 0; i < ELEMS_PER_THREAD; ++i)
    {

        auto logical_col = TILE_X * tileBlockIdx_x + tile_x_index + i * TILE_Y;
        auto logical_row = TILE_X * tileBlockIdx_y + tile_y_index;

        auto idx0 = logical_col;
        auto idx1 = logical_row;
        if(DIM != TransposeDim2)
            idx1 %= length1;
        auto idx2 = DIM == TransposeDim2 ? 0 : logical_row / length1;

        if(!TILE_ALIGNED && (idx0 >= length0 || idx1 >= length1 || idx2 >= length2))
            break;

        auto global_write_idx
            = idx0 * stride_out0 + idx1 * stride_out1 + idx2 * stride_out2 + offset_out;
        Handler<T_O, cbtype>::write(output, global_write_idx, val[i], store_cb_fn, store_cb_data);
    }
}

ROCFFT_DEVICE_EXPORT void rocfft_internal_transpose_var2(const void* data_p, void* back_p)
{
    auto data = static_cast<const DeviceCallIn*>(data_p);

    auto length  = data->node->length;
    auto istride = data->node->inStride;
    auto ostride = data->node->outStride;

    // grid Y counts rows on dims Y+Z, sliced into tiles of TILE_X.
    // grid Z counts any dims beyond Y+Z, plus batch
    unsigned int gridYrows = length[1] * (length.size() > 2 ? length[2] : 1);
    auto         highdim   = std::min<size_t>(length.size(), 3);
    unsigned int gridZ     = std::accumulate(
        length.begin() + highdim, length.end(), data->node->batch, std::multiplies<unsigned int>());
    // grid/threads get assigned in the macro that decides which kernel function to use
    dim3 grid;
    dim3 threads;

    int twl = 0;

    if(data->node->large1D > (size_t)256 * 256 * 256 * 256)
        printf("large1D twiddle size too large error");
    else if(data->node->large1D > (size_t)256 * 256 * 256)
        twl = 4;
    else if(data->node->large1D > (size_t)256 * 256)
        twl = 3;
    // TODO- possibly using a smaller LargeTwdBase for transpose by large_twiddle_base
    else if(data->node->large1D > (size_t)256)
        twl = 2;
    else if(data->node->large1D > 0)
        twl = 1;

    int          dir    = data->node->direction;
    CallbackType cbtype = data->get_callback_type();

    // check the length along the fast output dimension to decide if
    // we should do diagonal block ordering
    size_t fastOut = data->node->length[1];
    // diagonal ordering only seems to help 2D cases, not 3D
    bool diagonal = (fastOut % 256) == 0 && (data->node->outStride[0] % 256 == 0)
                    && data->node->scheme == CS_KERNEL_TRANSPOSE;

    if(array_type_is_interleaved(data->node->inArrayType)
       && array_type_is_interleaved(data->node->outArrayType))
    {
        LAUNCH_TRANSPOSE_KERNEL(interleaved, interleaved);
    }
    else if(array_type_is_interleaved(data->node->inArrayType)
            && array_type_is_planar(data->node->outArrayType))
    {
        LAUNCH_TRANSPOSE_KERNEL(interleaved, planar);
    }
    else if(array_type_is_planar(data->node->inArrayType)
            && array_type_is_interleaved(data->node->outArrayType))
    {
        LAUNCH_TRANSPOSE_KERNEL(planar, interleaved);
    }
    else if(array_type_is_planar(data->node->inArrayType)
            && array_type_is_planar(data->node->outArrayType))
    {
        LAUNCH_TRANSPOSE_KERNEL(planar, planar);
    }
    else
    {
        throw std::runtime_error("unhandled transpose array formats");
    }
}
