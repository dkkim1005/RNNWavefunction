#pragma once

template <typename FloatType>
TFIChain<FloatType>::TFIChain(const FloatType J, const FloatType h):
  J_(J),
  h_(h),
  options_(torch::TensorOptions()
    .dtype(TorchPrec<FloatType>::dtype)
    .device(torch::kCUDA)
    )
  {}

template <typename FloatType>
void TFIChain<FloatType>::calculate_diag_elem_(const int gpuBlockSize,
  const int nInputs,
  const int nBatches,
  const FloatType * spinStates_ptr,
  FloatType * htilda_ptr)
{
  gpu_kernel::calculate_diag_elem<<<gpuBlockSize, NUM_THREADS_PER_BLOCK>>>(nInputs,
    nBatches, J_, spinStates_ptr, htilda_ptr);
}


namespace gpu_kernel
{
template <typename FloatType>
__global__ void calculate_diag_elem(const int64_t nInputs,
  const int64_t nBatches,
  const FloatType J,
  const FloatType * spinStates_ptr,
  FloatType * htilda_ptr)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nBatches)
  {
    htilda_ptr[idx] = 0.0;
    for (int i=0; i<nInputs-1; ++i)
      htilda_ptr[idx] += spinStates_ptr[i*nBatches+idx]*spinStates_ptr[(i+1)*nBatches+idx];
    htilda_ptr[idx] += spinStates_ptr[idx]*spinStates_ptr[(nInputs-1)*nBatches+idx];
    htilda_ptr[idx] *= J;
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void get_psi_ratios(const int nBatches,
  const FloatType h, const FloatType * lnpsi1, const FloatType * lnpsi0,
  FloatType * htilda)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nBatches)
  {
    htilda[idx] += h*std::exp(lnpsi1[idx]-lnpsi0[idx]);
    idx += nstep;
  }
}
} // end namespace gpu_kernel
