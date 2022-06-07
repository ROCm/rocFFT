Runtime Compilation Design Document for rocFFT
==============================================

Copyright and Disclaimer
------------------------

DISCLAIMER

The information contained herein is for informational purposes only,
and is subject to change without notice. While every precaution has
been taken in the preparation of this document, it may contain
technical inaccuracies, omissions and typographical errors, and AMD is
under no obligation to update or otherwise correct this information.
Advanced Micro Devices, Inc. makes no representations or warranties
with respect to the accuracy or completeness of the contents of this
document, and assumes no liability of any kind, including the implied
warranties of noninfringement, merchantability or fitness for
particular purposes, with respect to the operation or use of AMD
hardware, software or other products described herein.  No license,
including implied or arising by estoppel, to any intellectual property
rights is granted by this document.  Terms and limitations applicable
to the purchase or use of AMD’s products are as set forth in a signed
agreement between the parties or in AMD's Standard Terms and
Conditions of Sale.

AMD is a trademark of Advanced Micro Devices, Inc.  Other product names
used in this publication are for identification purposes only and may
be trademarks of their respective companies.

Copyright (c) 2022 - present Advanced Micro Devices, Inc. All rights
reserved.

Summary
-------

This document describes runtime compilation (RTC) as it is used in
rocFFT.  Runtime compilation helps reduce binary size and build times
for the library, and can allow for optimizations that are not
practical versus ahead-of-time compiled kernels.

Problem
-------

Stockham FFT kernels make up the vast majority of the rocFFT library.
Kernels handling specific problem sizes are chosen as part of the
rocFFT build process, and are compiled for all of the variants that
might be required at runtime.

The count of variants for each problem size has a number of stacking
multipliers applied to it.  Any given problem size needs variants for:

* Each supported GPU architecture;

* Six interleaved/planar and in-place/out-of-place variants;

* Forward and inverse transforms;

* At least two precisions (single/double);

* Unit- and non-unit- strides;

* With and without callback support.

Runtime compilation has advantages over pre-compiling all of the
above variants for all problem sizes.  These include:

* Handling of new problem sizes does not require rebuilding the
  library.

* Build times are faster.

* The library binary is smaller.  This in turn means:
  installation is faster; and difficulties arising from limited
  memory addressing in shared objects (the default memory model for
  shared objects built with ``-fPIC`` only allows for 2 GiB binaries,
  resulting in `build breaks`_) are reduced.

.. _build breaks: https://www.ibm.com/support/pages/intel-compiler-error-relocation-truncated-fit-rx8664pc32

Solution
--------

HIP provides runtime compilation facilities, in the hiprtc.h header.

The code generator is embedded into the library itself.  During FFT
planning, we can run the code generator to produce suitable FFT
source code for the specific problem being solved.  Then, we can
runtime-compile that source into GPU kernels that we launch at FFT
execution time.

Empirical testing of runtime compilation on our FFT kernels shows
that compilation times for single kernels range between 0.5 and 2
seconds on modern hardware, with more complicated kernels taking more
time.  Kernel execution time is identical to the ahead-of-time
compiled version.

Implementation
--------------

Embedding and running the generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A generator implemented in C++ can be built into the library like any
other C++ code that makes up the library.

During plan building, we can execute the generator code to
produce a string of source code for the required problem sizes.

The generator needs sufficient input to have it produce exactly the
variant that is required, e.g. length-336, inverse, out-of-place,
interleaved-to-planar, double precision, etc.

Compilation
^^^^^^^^^^^

Compilation should be done during plan building, and the generated
kernels can be attached directly to the ``TreeNode`` for that step of
the FFT.

If the kernels are available on the ``TreeNode``, we will have less
overhead at execution time, since we don't need to do any work to
find the right kernel to run.

Caching kernels
^^^^^^^^^^^^^^^

If a process needs to create multiple plans that would compile the
same FFT kernel variant, it's nice to reuse kernels we've already
compiled.  Reusing an already-compiled kernel would save compilation
time on subsequent runs.

Compiled kernels can be persisted to disk for maximal reuse.  The
cache keys need to be chosen carefully to ensure that an obsolete
kernel is not reused when a new version really **should** be
recompiled.

The kernel function name shall be the main key field in the cache.
The function name shall encode all of the parameters by which kernel
functions could differ, including:

* scheme (e.g. whether this is a standard Stockham kernel, or a
  variant that does different read/write tiling)

* length (typically 1D length, may be 2D or more for single kernels
  that handle higher dimension transforms)

* placement (in-place or out-of-place)

* direction (forward or backward)

* input and output formats (planar, interleaved, real)

* precision (single or double)

* stride type (unit stride or non-unit stride)

* large 1D twiddle type and base (for kernels that need to integrate a
  twiddle multiplication step)

* callback type (run callbacks or don't run callbacks)

Encoding all of these parameters into the kernel name is necessary
anyways, so that logs and profilers will tell users and developers
exactly which kernel is running, even if it's been runtime-compiled.

Using just the kernel name as the main key is also helpful because
the caching code needn't be aware of all the possible parameters that
kernels could differ by.  New parameters can be added at anytime, and
as long as the kernel names are updated accordingly, the cache will
just work.

The cache will also need to store other key fields to ensure that a
kernel is compiled if any of these changes:

* GPU architecture

* HIP runtime version

* Kernel generator version

Practically, these key field choices will ensure that users are
always running the latest kernels that rocFFT provides and which are
appropriate for the hardware present.

User control of cache
:::::::::::::::::::::

Distributed workflows will want additional control over the cache.
For example, a workload that distributes FFT computation over a large
number of MPI nodes will want to ensure that the kernels are built
once centrally rather than by each node.

MPI nodes might also have no access to disk (either shared with other
nodes or local to each node).

rocFFT needs to expose APIs to:

* Serialize the current cache to a library-allocated buffer

* Free the library-allocated serialization buffer

* Deserialize a buffer into a cache (which might need to be in-memory
  for diskless nodes)

The example MPI computation described above would be able to build
plans on the rank 0 node to populate the cache once.  Then, it can
use these new APIs along with MPI APIs to distribute the cache to
each work node.

Backing store implementation
::::::::::::::::::::::::::::

The cache needs to be written to disk, and be robust in the face of
concurrent access, crashes during library operation, and so on.

We really would like the cache to have ACID properties of database
systems.

The easiest way to achieve this is to use SQLite to manage the
storage.  It's easily embeddable in our library (or is readily
available as its own library), and provides all of the properties
we'd want for the storage backend.

It also provides APIs to serialize a database, as required for the
distributed workflows described above.

Pre-built kernels
:::::::::::::::::

Even if rocFFT is prepared to runtime-compile any FFT kernel, we can
still pre-compile kernels by populating a cache at library build time
and shipping the cache with the library.

The main challenge here is installing this pre-built cache in a place
that the library will be able to find.

The easiest solution here, as employed by `other math libraries` is
to look for this the cache file next to the shared library itself.

.. _other math libraries: https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/d8e00e169ccc7ca21211705643e85545e98e455a/library/src/tensile_host.cpp#L521

Arbitrary criteria can be used to decide whether to pre-build
kernels, but for example the most common architectures, array
formats, and problem sizes might be pre-built.  Less common choices
could be runtime-compiled.

Impact on tests
:::::::::::::::

Accuracy tests are maximally affected in terms of runtime by this
change, since they run a huge number of problem sizes in the context
of a single process.  That means the costs of generating and
compiling a large variety of kernel variants will be the most painful
here, once more problem sizes are handled by the new generator.

An increase in test runtime is an unfortunate side effect of runtime
compilation.  This cost is made more acceptable because the compile
time of the library has already been reduced prior to running the tests.

A possible solution here might be to do a parallel traversal of the
test cases, building rocFFT plans for each of them (but not actually
executing plans).  This would runtime-compile the whole suite's
kernels in parallel, which would save a lot of time.

Interaction with callbacks
^^^^^^^^^^^^^^^^^^^^^^^^^^

Callback-enabled FFTs require a different kernel variant to be
generated, but the decision of whether to actually run with a
callback is made by the user after the plan is constructed.

To solve this, we generate both a callback and non-callback variant
where necessary during plan creation.

Parallel compilation
^^^^^^^^^^^^^^^^^^^^

Because of the potential need for callback-enabled kernels, most
plans will be generated faster if kernels can be compiled in
parallel.  Unfortunately, hipRTC has process-wide locks in it that
prevent useful multithreading of compilation.

Instead, we can spawn a helper process for subsequent compilations if
a compilation is already in-progress in the original process.  This
helper would need to be shipped with the library, in a location
that's knowable by the library.  If we fail to find or spawn that
helper, compilation must fall back to compiling in-process.

Future work
-----------

Moving away from chosen problem sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the infrastructure is in place, we could consider enabling
runtime compilation for all FFT sizes, not just those that are chosen
ahead of time.  The generator is already able to auto-factorize
arbitrary sizes, though we haven't yet tested the limits of this
ability.

Non-Stockham kernels
^^^^^^^^^^^^^^^^^^^^

Aside from the Stockham kernels included in this proposal, rocFFT
includes other kernels:

* Bluestein
* transpose
* C2R/R2C pre/post processing

They could also be compiled at runtime.

