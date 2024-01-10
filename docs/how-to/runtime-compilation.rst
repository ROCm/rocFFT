.. meta::
  :description: rocFFT documentation and API reference library
  :keywords: rocFFT, ROCm, API, documentation

.. _runtime-compilation:

********************************************************************
Runtime Compilation
********************************************************************

Runtime compilation
===================

rocFFT includes many kernels for common FFT problems.  Some plans may
require additional kernels aside from what is built in to the
library.  In these cases, rocFFT will compile optimized kernels for
the plan when the plan is created.

Compiled kernels are stored in memory by default and will be reused
if they are required again for plans in the same process.

If the ``ROCFFT_RTC_CACHE_PATH`` environment variable is set to a
writable file location, rocFFT will write compiled kernels to this
location.  rocFFT will read kernels from this location for plans in
other processes that need runtime-compiled kernels.  rocFFT will
create the specified file if it does not already exist.
