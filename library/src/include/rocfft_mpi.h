// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_MPI_H
#define ROCFFT_MPI_H

#ifdef ROCFFT_MPI_ENABLE
#include <mpi.h>
#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>
#endif

class MPI_Comm_wrapper_t
{
public:
    MPI_Comm_wrapper_t() = default;

    // conversion to unwrapped communicator for passing to MPI APIs
    operator MPI_Comm() const
    {
        return mpi_comm;
    }

    // copy, duplicating the communicator
    MPI_Comm_wrapper_t(const MPI_Comm_wrapper_t& other)
    {
        duplicate(other.mpi_comm);
    }
    MPI_Comm_wrapper_t& operator=(const MPI_Comm_wrapper_t& other)
    {
        duplicate(other.mpi_comm);
        return *this;
    }

    // move communicator
    MPI_Comm_wrapper_t(MPI_Comm_wrapper_t&& other)
    {
        std::swap(this->mpi_comm, other.mpi_comm);
    }
    MPI_Comm_wrapper_t& operator=(MPI_Comm_wrapper_t&& other)
    {
        std::swap(this->mpi_comm, other.mpi_comm);
        return *this;
    }

    ~MPI_Comm_wrapper_t()
    {
        free();
    }

    void free()
    {
        if(mpi_comm != MPI_COMM_NULL)
            MPI_Comm_free(&mpi_comm);
        mpi_comm = MPI_COMM_NULL;
    }

    void duplicate(MPI_Comm in_comm)
    {
        free();
        if(in_comm != MPI_COMM_NULL && MPI_Comm_dup(in_comm, &mpi_comm) != MPI_SUCCESS)
        {
            throw std::runtime_error("failed to duplicate MPI communicator");
        }
    }

    // check if communicator has been initialized
    operator bool() const
    {
        return mpi_comm != MPI_COMM_NULL;
    }
    bool operator!() const
    {
        return mpi_comm == MPI_COMM_NULL;
    }

private:
    MPI_Comm mpi_comm = MPI_COMM_NULL;
};
#endif

#endif
