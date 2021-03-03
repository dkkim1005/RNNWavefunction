#pragma once

#include <vector>
#include <torch/torch.h>
#include "common.cuh"
#include "trng4cuda.cuh"

// positive-valued wavefunction with 1-layer GRU
template <typename FloatType>
class pRNN : public torch::nn::Module
{
public:
  pRNN(const int64_t nInputs, // length of a spin array
    const int64_t nHiddens, // # of memory cells
    const int64_t nBatches, // # of spin arrays
    const unsigned long seedNumber, // random seed
    const unsigned long seedDistance); // size of seed block
  // return lnpsi and spin states
  std::vector<at::Tensor> forward();
  // return lnpsi for given spin states (grad is not calculated in the scope.)
  at::Tensor forward_no_grad(const at::Tensor & spinStates);
  // return lnpsi for given spin states
  at::Tensor forward_with_grad(const at::Tensor & spinStates);
  int64_t get_nInputs() const { return knInputs; }
  int64_t get_nBatches() const { return knBatches; }

private:
  const int64_t knInputs, knHiddens, knBatches, kgpuBlockSize;
  torch::nn::GRUCell GRULayer_;
  torch::nn::Linear LinearLayer_;
  torch::nn::Softmax SoftmaxLayer_;
  const at::Tensor s0_, h0_;
  at::Tensor y_tmp_, sigma_tmp_;
  TRNGWrapper<FloatType, trng::yarn2> randDev_;
  thrust::device_vector<FloatType> randVal_dev_;
  torch::TensorOptions options_;
};


namespace gpu_kernel
{
template <typename FloatType>
__global__ void sampling_states(const int nBatches, const FloatType * y, const FloatType * randVal,
  FloatType * spinState, FloatType * sigma);

template <typename FloatType>
__global__ void set_sigma(const int nBatches, const FloatType * spinState, FloatType * sigma);
} // end namespace gpu_kernel


#include "impl_rnnwavefunction.cuh"
