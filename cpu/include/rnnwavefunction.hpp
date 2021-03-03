#pragma once

#include <vector>
#include <torch/torch.h>
#include <trng/yarn2.hpp>
#include <trng/uniform01_dist.hpp>
#include "common.hpp"

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
  // return lnpsi for given spin states
  at::Tensor forward(const at::Tensor & spinStates);
  int64_t get_nInputs() const { return knInputs; }
  int64_t get_nBatches() const { return knBatches; }

private:
  void sampling_states_(const FloatType * y, FloatType * spinState, FloatType * sigma);
  void set_sigma_(const FloatType * spinState, FloatType * sigma) const;

  const int64_t knInputs, knHiddens, knBatches;
  torch::nn::GRUCell GRULayer_;
  torch::nn::Linear LinearLayer_;
  torch::nn::Softmax SoftmaxLayer_;
  const at::Tensor s0_, h0_;
  at::Tensor y_tmp_, sigma_tmp_;
  std::vector<trng::yarn2> randDev_;
  trng::uniform01_dist<FloatType> randUniform_;
  torch::TensorOptions options_;
};


#include "impl_rnnwavefunction.hpp"
