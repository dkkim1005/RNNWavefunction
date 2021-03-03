#pragma once

#include <vector>
#include <torch/torch.h>
#include <cmath>
#include "common.cuh"

namespace gpu_kernel
{
template <typename FloatType>
__global__ void calculate_diag_elem(const int64_t nInputs,
  const int64_t nBatches,
  const FloatType J,
  const FloatType * spinStates_ptr,
  FloatType * htilda_ptr);

template <typename FloatType>
__global__ void get_psi_ratios(const int nBatches,
  const FloatType h, const FloatType * lnpsi1, const FloatType * lnpsi0,
  FloatType * htilda);
} // end namespace gpu_kernel


template <typename FloatType>
class TFIChain
{
public:
  // J : coupling constatnt, h : transverse-field strength
  TFIChain(const FloatType J, const FloatType h);
  // return {loss, htilda}
  template <typename AnsatzType>
  std::vector<at::Tensor> construct_loss(AnsatzType & autoRegAnsatz)
  {
    const int gpuBlockSize = CHECK_BLOCK_SIZE(1+(autoRegAnsatz.get_nBatches()-1)/NUM_THREADS_PER_BLOCK);
    // res[0] : lnpsi, res[1] : spinStates
    auto res = autoRegAnsatz.forward();
    at::Tensor & lnpsi = res[0], & spinStates = res[1];
    // diag = \sum_{i} J*spinStates_{i}*spinStates_{i+1}
    auto htilda = torch::zeros({autoRegAnsatz.get_nBatches()}, options_);
    this->calculate_diag_elem_(gpuBlockSize,
      autoRegAnsatz.get_nInputs(),
      autoRegAnsatz.get_nBatches(),
      spinStates.data_ptr<FloatType>(),
      htilda.template data_ptr<FloatType>());
    // htilda = diag + h*\sum_{i} exp(lnpsi(..., -s, ...)-lnpsi(..., s, ...))
    this->calculate_off_elem_(gpuBlockSize,
      lnpsi.data_ptr<FloatType>(),
      spinStates,
      autoRegAnsatz,
      htilda.template data_ptr<FloatType>());
    // grad : \partial_i [2*(<htilda*lnpsi> -<htilda>*<lnpsi>)], energy : <htilda>
    return {2.0*(lnpsi*(htilda-htilda.mean())).mean(), htilda};
  }

private:
  // shape of spinStates : [nInputs, nBatches]
  void calculate_diag_elem_(const int gpuBlockSize, const int nInputs, const int nBatches, const FloatType * spinStates_ptr, FloatType * htilda_ptr);

  template <typename AnsatzType>
  void calculate_off_elem_(const int gpuBlockSize, const FloatType * lnpsi0_ptr, const at::Tensor & spinStates, AnsatzType & autoRegAnsatz, FloatType * htilda_ptr)
  {
    for (int i=0; i<autoRegAnsatz.get_nInputs(); ++i)
    {
      FloatType * spinStates_ptr = spinStates[i].data_ptr<FloatType>();
      // flip spins on the i site
      gpu_kernel::common__ScalingVector__<<<gpuBlockSize, NUM_THREADS_PER_BLOCK>>>(-1.0,
        autoRegAnsatz.get_nBatches(), spinStates_ptr);
      const at::Tensor & lnpsi1 = autoRegAnsatz.forward_no_grad(spinStates);
      const FloatType * lnpsi1_ptr = lnpsi1.data_ptr<FloatType>();
      gpu_kernel::get_psi_ratios<<<gpuBlockSize, NUM_THREADS_PER_BLOCK>>>(autoRegAnsatz.get_nBatches(),
        h_, lnpsi1_ptr, lnpsi0_ptr, htilda_ptr);
      // reflip spins on the i site
      gpu_kernel::common__ScalingVector__<<<gpuBlockSize, NUM_THREADS_PER_BLOCK>>>(-1.0,
        autoRegAnsatz.get_nBatches(), spinStates_ptr);
    }
    const FloatType InvnInputs = 1.0/static_cast<FloatType>(autoRegAnsatz.get_nInputs());
    gpu_kernel::common__ScalingVector__<<<gpuBlockSize, NUM_THREADS_PER_BLOCK>>>(InvnInputs,
      autoRegAnsatz.get_nBatches(), htilda_ptr);
  }

  const FloatType J_, h_;
  torch::TensorOptions options_;
};


#include "impl_hamiltonians.cuh"
