/*******************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

template <typename T>
__device__ void FwdRad4B1(T* R0, T* R2, T* R1, T* R3)
{

    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);

    (*R3) = (*R1) + T(-(*R3).y, (*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);

    res   = (*R1);
    (*R1) = (*R2);
    (*R2) = res;
}

template <typename T>
__device__ void InvRad4B1(T* R0, T* R2, T* R1, T* R3)
{

    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + T((*R3).y, -(*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);

    res   = (*R1);
    (*R1) = (*R2);
    (*R2) = res;
}
