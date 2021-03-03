#pragma once

template <typename FloatType>
TFIChain<FloatType>::TFIChain(const FloatType J, const FloatType h):
  J_(J),
  h_(h),
  options_(torch::TensorOptions().dtype(TorchPrec<FloatType>::dtype)) {}


template <typename FloatType>
void TFIChain<FloatType>::calculate_diag_elem_(const int64_t nInputs,
  const int64_t nBatches,
  const FloatType * spinStates_ptr,
  FloatType * htilda_ptr)
{
  for (int k=0; k<nBatches; ++k)
  {
    htilda_ptr[k] = 0;
    for (int i=0; i<nInputs-1; ++i)
      htilda_ptr[k] += spinStates_ptr[i*nBatches+k]*spinStates_ptr[(i+1)*nBatches+k];
    htilda_ptr[k] += spinStates_ptr[k]*spinStates_ptr[(nInputs-1)*nBatches+k];
    htilda_ptr[k] *= J_;
  }
}
