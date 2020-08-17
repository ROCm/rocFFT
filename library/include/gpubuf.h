#ifndef ROCFFT_GPUBUF_H
#define ROCFFT_GPUBUF_H

#include <hip/hip_runtime_api.h>

// Simple RAII class for GPU buffers.  T is the type of pointer that
// data() returns
template <class T = void>
class gpubuf_t
{
public:
    gpubuf_t()
        : buf(nullptr)
    {
    }
    // buffers are movable but not copyable
    gpubuf_t(gpubuf_t&& other)
    {
        std::swap(buf, other.buf);
    }
    gpubuf_t& operator=(gpubuf_t&& other)
    {
        std::swap(buf, other.buf);
        return *this;
    }
    gpubuf_t(const gpubuf_t&) = delete;
    gpubuf_t& operator=(const gpubuf_t&) = delete;

    ~gpubuf_t()
    {
        free();
    }

    hipError_t alloc(const size_t size)
    {
        free();
        auto ret = hipMalloc(&buf, size);
        if(ret != hipSuccess)
            buf = nullptr;
        return ret;
    }

    void free()
    {
        if(buf != nullptr)
        {
            hipFree(buf);
            buf = nullptr;
        }
    }

    T* data() const
    {
        return static_cast<T*>(buf);
    }

    // equality/bool tests
    bool operator==(std::nullptr_t n) const
    {
        return buf == n;
    }
    bool operator!=(std::nullptr_t n) const
    {
        return buf != n;
    }
    operator bool() const
    {
        return buf;
    }

private:
    // The GPU buffer
    void* buf;
};

// default gpubuf that gives out void* pointers
typedef gpubuf_t<> gpubuf;
#endif
