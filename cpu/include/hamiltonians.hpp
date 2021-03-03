#pragma once

#include <vector>
#include <torch/torch.h>
#include <cmath>
#include "common.hpp"

template <typename FloatType>
class TFIChain
{
public:
  // J : coupling constatnt, h : transverse-field strength
  TFIChain(const FloatType J, const FloatType h);
  // return {loss, energy}
  template <typename AnsatzType>
  std::vector<at::Tensor> construct_loss(AnsatzType & autoRegAnsatz)
  {
    // res[0] : lnpsi, res[1] : spinStates
    auto res = autoRegAnsatz.forward();
    at::Tensor & lnpsi = res[0], & spinStates = res[1];
    // diag = \sum_{i} J*spinStates_{i}*spinStates_{i+1}
    auto htilda = torch::zeros({autoRegAnsatz.get_nBatches()}, options_);
    this->calculate_diag_elem_(autoRegAnsatz.get_nInputs(),
      autoRegAnsatz.get_nBatches(),
      spinStates.data_ptr<FloatType>(),
      htilda.template data_ptr<FloatType>());
    // htilda = diag + h*\sum_{i} exp(lnpsi(..., -s, ...)-lnpsi(..., s, ...))
    this->calculate_off_elem_(lnpsi.data_ptr<FloatType>(),
      spinStates,
      autoRegAnsatz,
      htilda.template data_ptr<FloatType>());
    // grad : \partial_i [2*(<htilda*lnpsi> -<htilda>*<lnpsi>)], energy : <htilda>
    return {2.0*(lnpsi*(htilda-htilda.mean())).mean(), htilda};
  }

private:
  // shape of spinStates : [nInputs, nBatches]
  void calculate_diag_elem_(const int64_t nInputs, const int64_t nBatches, const FloatType * spinStates_ptr, FloatType * htilda_ptr);

  template <typename AnsatzType>
  void calculate_off_elem_(const FloatType * lnpsi0_ptr, const at::Tensor & spinStates, AnsatzType & autoRegAnsatz, FloatType * htilda_ptr)
  {
    for (int i=0; i<autoRegAnsatz.get_nInputs(); ++i)
    {
      FloatType * spinStates_ptr = spinStates[i].data_ptr<FloatType>();
      // flip spins on the i site
      for (int k=0; k<autoRegAnsatz.get_nBatches(); ++k)
        spinStates_ptr[k] *= -1.0;
      const at::Tensor & lnpsi1 = autoRegAnsatz.forward(spinStates);
      const FloatType * lnpsi1_ptr = lnpsi1.data_ptr<FloatType>();
      for (int k=0; k<autoRegAnsatz.get_nBatches(); ++k)
        htilda_ptr[k] += h_*std::exp(lnpsi1_ptr[k]-lnpsi0_ptr[k]);
      // reflip spins on the i site
      for (int k=0; k<autoRegAnsatz.get_nBatches(); ++k)
        spinStates_ptr[k] *= -1.0;
    }
    const FloatType InvnInputs = 1.0/static_cast<FloatType>(autoRegAnsatz.get_nInputs());
    for (int k=0; k<autoRegAnsatz.get_nBatches(); ++k)
      htilda_ptr[k] *= InvnInputs;
  }

  const FloatType J_, h_;
  torch::TensorOptions options_;
};


#include "impl_hamiltonians.hpp"
