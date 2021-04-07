# Change Log for rocFFT

Full documentation for rocFFT is available at [rocfft.readthedocs.io](https://rocfft.readthedocs.io/en/latest/).

## [(Unreleased) rocFFT 1.0.12 for ROCm 4.3.0]

### Removed
- Removed R2C pair schemes and kernels.

### Optimizations
- Optimized 2D/3D R2C 100 and 1D Z2Z 2500.

### Fixed
- Fixed potential crashes in 3D transforms with unusual strides, for
  SBCC-optimized sizes.

## [(Unreleased) rocFFT 1.0.11 for ROCm 4.2.0]

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

## [rocFFT 1.0.10 for ROCm 4.1.0]

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

## [rocFFT 1.0.8 for ROCm 3.10.0]

### Optimizations
- Optimized 1D length 10000 C2C case.

### Changed
- Added BUILD_CLIENTS_ALL CMake option.

### Fixed
- Fixed correctness of SBCC/SBRC kernels with non-unit strides.
- Fixed fused C2R kernel when a Bluestein transform follows it.

## [rocFFT 1.0.7 for ROCm 3.9.0]

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

## [rocFFT 1.0.6 for ROCm 3.8.0]

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

## [rocFFT 1.0.5 for ROCM 3.7.0]

### Optimizations
- Optimized C2C power-of-2 middle sizes.

### Changed
- Parallelized work in unit tests and eliminate duplicate cases.

### Fixed
- Fixed correctness of certain large 1D, and 2D power-of-3, 5 sizes.
- Fixed incorrect buffer assignment for some even-length R2C transforms.
- Fixed `<cstddef>` inclusion on C compilers.
- Fixed incorrect results on non-unit strides with SBCC/SBRC kernels.
