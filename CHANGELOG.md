# Change Log for rocFFT

Full documentation for rocFFT is available at [rocfft.readthedocs.io](https://rocfft.readthedocs.io/en/latest/).

## [(Unreleased) rocFFT 1.0.9 for ROCm 4.0.0]

### Added

- Explicitly specify MAX_THREADS_PER_BLOCK through `__launch_bounds__` for all
  kernels.

- New radix-11 and radix-13 kernels; used in length 11 and 13 (and
  some of their multiples) transforms.

### Changed

- rocFFT now automatically allocates a work buffer if the plan
  requires one but none is provided

- An explicit `rocfft_status_invalid_work_buffer` error is now
  returned when a work buffer of insufficient size is
  explicitly provided.
