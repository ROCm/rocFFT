# Change Log for rocFFT

Full documentation for rocFFT is available at [rocfft.readthedocs.io](https://rocfft.readthedocs.io/en/latest/).
 
## [(Unreleased) rocFFT 1.0.9 for ROCm 4.0.0]

### Changed

- rocFFT now automatically allocates a work buffer if the plan
  requires one but none is provided

- An explicit `rocfft_status_invalid_work_buffer` error is now
  returned when a work buffer of insufficient size is
  explicitly provided.
