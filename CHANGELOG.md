# Change Log for rocFFT

Full documentation for rocFFT is available at [rocfft.readthedocs.io](https://rocfft.readthedocs.io/en/latest/).

## rocFFT 1.0.25 for ROCm 6.0.0

### Optimizations
- Improved performance of some small 2D/3D real FFTs supported by 2D_SINGLE kernel.

### Changed
- Built kernels in solution-map to library kernel cache.

- rocfft-rider and dyna-rocfft-rider have been renamed to rocfft-bench and dyna-rocfft-bench, controlled by the
  BUILD_CLIENTS_BENCH CMake option.  Links for the old file names are installed, and the old
  BUILD_CLIENTS_RIDER CMake option is accepted for compatibility but both will be removed in a future release.

## rocFFT 1.0.24 for ROCm 5.7.0

### Optimizations
- Improved performance of complex forward/inverse 1D FFTs (2049 <= length <= 131071) that use Bluestein's algorithm.

### Added
- Implemented a solution map version converter and finish the first conversion from ver.0 to ver.1. Where version 1 removes some incorrect kernels (sbrc/sbcr using half_lds)

### Changed

- Moved rocfft_rtc_helper executable to lib/rocFFT directory on Linux.
- Moved library kernel cache to lib/rocFFT directory.

## rocFFT 1.0.23 for ROCm 5.6.0

### Added
- Implemented half-precision transforms, which can be requested by passing rocfft_precision_half to rocfft_plan_create.
- Implemented a hierarchical solution map which saves how to decompose a problem and the kernels to be used.
- Implemented a first version of offline-tuner to support tuning kernels for C2C/Z2Z problems.

### Changed
- Replaced std::complex with hipComplex data types for data generator.
- FFT plan dimensions are now sorted to be row-major internally where possible, which produces better plans if the dimensions were accidentally specified in a different order (column-major, for example).
- Added --precision argument to benchmark/test clients.  --double is still accepted but is deprecated as a method to request a double-precision transform.
- Improved performance test suite statistical framework.

### Fixed
- Fixed over-allocation of LDS in some real-complex kernels, which was resulting in kernel launch failure.

## rocFFT 1.0.22 for ROCm 5.5.0

### Optimizations
- Improved performance of 1D lengths < 2048 that use Bluestein's algorithm.
- Reduced time for generating code during plan creation.
- Optimized 3D R2C/C2R lengths 32, 84, 128.
- Optimized batched small 1D R2C/C2R cases.

### Added
- Added gfx1101 to default AMDGPU_TARGETS.

### Changed
- Moved client programs to C++17.
- Moved planar kernels and infrequently used Stockham kernels to be runtime-compiled.
- Moved transpose, real-complex, Bluestein, and Stockham kernels to library kernel cache.

### Fixed
- Removed zero-length twiddle table allocations, which fixes errors from hipMallocManaged.
- Fixed incorrect freeing of HIP stream handles during twiddle computation when multiple devices are present.

## rocFFT 1.0.21 for ROCm 5.4.3

### Fixed
- Removed source directory from rocm_install_targets call to prevent installation of rocfft.h in an unintended location.

## rocFFT 1.0.20 for ROCm 5.4.1

### Fixed
- Fixed incorrect results on strided large 1D FFTs where batch size does not equal the stride.

## rocFFT 1.0.19 for ROCm 5.4.0

### Optimizations
- Optimized some strided large 1D plans.

### Added
- Added rocfft_plan_description_set_scale_factor API to efficiently multiply each output element of a FFT by a given scaling factor.
- Created a rocfft_kernel_cache.db file next to the installed library. SBCC/CR/RC kernels are moved to this file when built with the library, and are runtime-compiled for new GPU architectures.
- Added gfx1100 and gfx1102 to default AMDGPU_TARGETS.

### Changed
- Moved runtime compilation cache to in-memory by default.  A default on-disk cache can encounter contention problems
on multi-node clusters with a shared filesystem.  rocFFT can still be told to use an on-disk cache by setting the
ROCFFT_RTC_CACHE_PATH environment variable.

## rocFFT 1.0.18 for ROCm 5.3.0

### Changed
- Runtime compilation cache now looks for environment variables XDG_CACHE_HOME (on Linux) and LOCALAPPDATA (on
  Windows) before falling back to HOME.
- Moved computation of the twiddle table from host to the device.

### Optimizations
- Optimized 2D R2C/C2R to use 2-kernel plans where possible.
- Improved performance of the Bluestein algorithm.
- Optimized sbcc-168 and 100 by using half-lds.
- Optimized length-280 2D/3D transforms.
- Added kernels for factorizable 1D lengths < 128

### Fixed
- Fixed occasional failures to parallelize runtime compilation of kernels.
  Failures would be retried serially and ultimately succeed, but this would take extra time.
- Fixed failures of some R2C 3D transforms that use the unsupported TILE_UNALGNED SBRC kernels.
  An example is 98^3 R2C out-of-place.
- Fixed bugs in SBRC_ERC type.

## rocFFT 1.0.17 for ROCm 5.2.0
### Added
- Packages for test and benchmark executables on all supported OSes using CPack.
- Added File/Folder Reorg Changes with backward compatibility support using ROCM-CMAKE wrapper functions.

### Changed
- Improved reuse of twiddle memory between plans.
- Set a default load/store callback when only one callback
  type is set via the API for improved performance.
- Updated googletest dependency to version 1.11.

### Optimizations
- Introduced a new access pattern of lds (non-linear) and applied it on
  sbcc kernels len 64 and 81 to get performance improvement.
- Applied lds-non-linear and direct-load-to-register on sbcr kernels to
  get performance improvement.
- Applied lds-non-linear and direct-store-from-register on sbrc kernels to
  get performance improvement.

### Fixed
- Fixed correctness of certain transforms with unusual strides.
- Fixed incorrect handling of user-specified stream for runtime-compiled kernels.
- Fixed incorrect buffer allocation in rocfft-test on in-place transforms with different input and output sizes.

## rocFFT 1.0.16 for ROCm 5.1.0

### Changed
- Supported unaligned tile dimension for SBRC_2D kernels.
- Improved (more RAII) test and benchmark infrastructure.
- Enabled runtime compilation of length-2304 FFT kernel during plan creation.
- Added tokenizer for test suite.
- Reduce twiddle memory requirements for even-length real-complex transforms.
- Clients can now be built separately from the main library.

### Optimizations
- Optimized more large 1D cases by using L1D_CC plan.
- Optimized 3D 200^3 C2R case.
- Optimized 1D 2^30 double precision on MI200.
- Added padding to work buffer sizes to improve performance in many cases.

### Fixed
- Fixed correctness of some R2C transforms with unusual strides.

### Removed
- The hipFFT API (header) has been removed from after a long deprecation period.  Please use the [hipFFT](https://github.com/ROCmSoftwarePlatform/hipFFT) package/repository to obtain the hipFFT API.

## rocFFT 1.0.15 for ROCm 5.0.0

### Changed
- Enabled runtime compilation of single FFT kernels > length 1024.
- Re-aligned split device library into 4 roughly equal libraries.
- Implemented the FuseShim framework to replace the original OptimizePlan
- Implemented the generic buffer-assignment framework. The buffer assignment
  is no longer performed by each node. We designed a generic algorithm to
  test and pick the best assignment path.
  With the help of FuseShim, we can achieve more kernel-fusions as possible.
- Do not read the imaginary part of the DC and Nyquist modes for even-length
  complex-to-real transforms.

### Optimizations
- Optimized twiddle-conjugation; complex-to-complex inverse transforms should have similar performance to foward transforms now.
- Improved performance of single-kernel small 2D transforms.

## rocFFT 1.0.14 for ROCm 4.5.0

### Optimizations
- Optimized SBCC kernels of length 52, 60, 72, 80, 84, 96, 104, 108, 112, 160,
  168, 208, 216, 224, 240 with new kernel generator.

### Added
- Added support for Windows 10 as a build target.

### Changed
- Packaging split into a runtime package called rocfft and a development package called rocfft-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.

### Fixed
- Fixed a few validation failures of even-length R2C inplace. 2D, 3D cubics sizes such as
  100^2 (or ^3), 200^2 (or ^3), 256^2 (or ^3)...etc. We don't combine the three kernels
  (stockham-r2c-transpose). We only combine two kernels (r2c-transpose) instead.

### Changed
- Split 2D device code into separate libraries.

## rocFFT 1.0.13 for ROCm 4.4.0

### Optimizations
- Improved many plans by removing unnecessary transpose steps.
- Optimized scheme selection for 3D problems.
  - Imposed less restrictions on 3D_BLOCK_RC selection. More problems can use 3D_BLOCK_RC and
    have some performance gain.
  - Enabled 3D_RC. Some 3D problems with SBCC-supported z-dim can use less kernels and get benefit.
  - Force --length 336 336 56 (dp) use faster 3D_RC to avoid it from being skipped by conservative
    threshold test.
- Optimized some even-length R2C/C2R cases by doing more operations
  in-place and combining pre/post processing into Stockham kernels.
- Added radix-17.

### Added
- Added new kernel generator for select fused-2D transforms.

### Fixed
- Improved large 1D transform decompositions.

## rocFFT 1.0.12 for ROCm 4.3.0

### Changed
  Re-split device code into single-precision, double-precision, and miscellaneous kernels.

### Fixed
- Fixed potential crashes in double-precision planar->planar transpose.
- Fixed potential crashes in 3D transforms with unusual strides, for
  SBCC-optimized sizes.
- Improved buffer placement logic.

### Added
- Added new kernel generator for select lengths.  New kernels have
  improved performance.
- Added public `rocfft_execution_info_set_load_callback` and
  `rocfft_execution_info_set_store_callback` API functions to allow
  executing extra logic when loading/storing data from/to global
  memory during a transform.

### Removed
- Removed R2C pair schemes and kernels.

### Optimizations
- Optimized 2D/3D R2C 100 and 1D Z2Z 2500.
- Reduced number of kernels for 2D/3D sizes where higher dimension is 64, 128, 256.

### Fixed
- Fixed potential crashes in 3D transforms with unusual strides, for
  SBCC-optimized sizes.

## rocFFT 1.0.11 for ROCm 4.2.0

### Changed
  Move device code into main library.

### Optimizations
- Improved performance for single precision kernels exercising all except radix-2/7 butterfly ops.
- Minor optimization for C2R 3D 100, 200 cube sizes.
- Optimized some C2C/R2C 3D 64, 81, 100, 128, 200, 256 rectangular sizes.
- When factoring, test to see if remaining length is explicitly
  supported.
- Explicitly add radix-7 lengths 14, 21, and 224 to list of supported
  lengths.
- Optimized R2C 2D/3D 128, 200, 256 cube sizes.

### Fixed
- Fixed potential crashes in small 3D transforms with unusual strides.
  (https://github.com/ROCmSoftwarePlatform/rocFFT/issues/311)
- Fixed potential crashes when executing transforms on multiple devices.
  (https://github.com/ROCmSoftwarePlatform/rocFFT/issues/310)

## rocFFT 1.0.10 for ROCm 4.1.0

### Added
- Explicitly specify MAX_THREADS_PER_BLOCK through _\_launch\_bounds\_ for all
  kernels.
- Switch to new syntax for specifying AMD GPU architecture names and features.

### Optimizations
- Optimized C2C/R2C 3D 64, 81, 100, 128, 200, 256 cube sizes.
- Improved performance of the standalone out-of-place transpose kernel.
- Optimized 1D length 40000 C2C case.
- Enabled radix-7 for size 336.
- New radix-11 and radix-13 kernels; used in length 11 and 13 (and some of their multiples) transforms.

### Changed
- rocFFT now automatically allocates a work buffer if the plan
  requires one but none is provided.
- An explicit `rocfft_status_invalid_work_buffer` error is now
  returned when a work buffer of insufficient size is provided.
- Updated online documentation.
- Updated debian package name version with separated '_'.
- Adjusted accuracy test tolerances and how they are compared.

### Fixed
- Fixed 4x4x8192 accuracy failure.

## rocFFT 1.0.8 for ROCm 3.10.0

### Optimizations
- Optimized 1D length 10000 C2C case.

### Changed
- Added BUILD_CLIENTS_ALL CMake option.

### Fixed
- Fixed correctness of SBCC/SBRC kernels with non-unit strides.
- Fixed fused C2R kernel when a Bluestein transform follows it.

## rocFFT 1.0.7 for ROCm 3.9.0

### Optimizations
- New R2C and C2R fused kernels to combine pre/post processing steps with transpose.
- Enabled diagonal transpose for 1D and 2D power-of-2 cases.
- New single kernels for small power-of-2, 3, 5 sizes.
- Added more radix-7 kernels.

### Changed
- Explicitly disable XNACK and SRAM-ECC features on AMDGPU hardware.

### Fixed
- Fixed 2D C2R transform with length 1 on one dimension.
- Fixed potential thread unsafety in logging.

## rocFFT 1.0.6 for ROCm 3.8.0

### Optimizations
- Improved performance of 1D batch-paired R2C transforms of odd length.
- Added some radix-7 kernels.
- Improved performance for 1D length 6561, 10000.
- Improved performance for certain 2D transform sizes.

### Changed
- Allow static library build with BUILD_SHARED_LIBS=OFF CMake option.
- Updated googletest dependency to version 1.10.

### Fixed
- Fixed correctness of certain large 2D sizes.

## rocFFT 1.0.5 for ROCM 3.7.0

### Optimizations
- Optimized C2C power-of-2 middle sizes.

### Changed
- Parallelized work in unit tests and eliminate duplicate cases.

### Fixed
- Fixed correctness of certain large 1D, and 2D power-of-3, 5 sizes.
- Fixed incorrect buffer assignment for some even-length R2C transforms.
- Fixed `<cstddef>` inclusion on C compilers.
- Fixed incorrect results on non-unit strides with SBCC/SBRC kernels.
