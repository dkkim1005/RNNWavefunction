#pragma once

template <typename FloatType>
pRNN<FloatType>::pRNN(const int64_t nInputs,
  const int64_t nHiddens,
  const int64_t nBatches,
  const unsigned long seedNumber,
  const unsigned long seedDistance):
  knInputs(nInputs),
  knHiddens(nHiddens),
  knBatches(nBatches),
  GRULayer_(register_module("GRU-1 Layer", torch::nn::GRUCell(1, nHiddens))),
  LinearLayer_(register_module("Softmax Layer", torch::nn::Linear(nHiddens, 2))),
  SoftmaxLayer_(1),
  s0_(torch::zeros({nBatches, 1},
    torch::TensorOptions().dtype(TorchPrec<FloatType>::dtype))),
  h0_(torch::zeros({nBatches, nHiddens},
    torch::TensorOptions().dtype(TorchPrec<FloatType>::dtype))),
  y_tmp_(torch::zeros({nInputs, nBatches, 2}, 
    torch::TensorOptions().dtype(TorchPrec<FloatType>::dtype))),
  sigma_tmp_(torch::zeros({nInputs, nBatches, 2}, 
    torch::TensorOptions().dtype(TorchPrec<FloatType>::dtype))),
  randDev_(nBatches),
  options_(
    torch::TensorOptions().dtype(TorchPrec<FloatType>::dtype))
{
  this->to(TorchPrec<FloatType>::dtype);
  // block splitting scheme for parallel Monte-Carlo
  for (int k=0; k<nBatches; ++k)
  {
    randDev_[k].seed(seedNumber);
    randDev_[k].jump(2*seedDistance*k);
  }
}

template <typename FloatType>
std::vector<at::Tensor> pRNN<FloatType>::forward()
{
  auto spinStates = torch::zeros({knInputs, knBatches, 1}, options_);
  // y[.][0] : probability to be 'up' state
  // y[.][1] : probability to be 'down' state
  auto y = torch::zeros({knInputs, knBatches, 2}, options_);
  // sigma[.] = {1, 0} or {0, 1}
  auto sigma = torch::zeros({knInputs, knBatches, 2}, options_);
  auto hiddens = GRULayer_(s0_, h0_);
  y[0] = SoftmaxLayer_(LinearLayer_(hiddens));
  this->sampling_states_(y[0].data_ptr<FloatType>(),
      spinStates[0].data_ptr<FloatType>(),
      sigma[0].data_ptr<FloatType>());
  // lnpsi : \sum_i log(sqrt(y_i*sigma_i))
  auto lnpsi = torch::zeros({knBatches}, options_);
  lnpsi += 0.5*torch::log(torch::amax(y[0]*sigma[0], 1));
  for (int i=1; i<knInputs; ++i)
  {
    hiddens = GRULayer_(spinStates[i-1], hiddens);
    y[i] = SoftmaxLayer_(LinearLayer_(hiddens));
    this->sampling_states_(y[i].data_ptr<FloatType>(),
        spinStates[i].data_ptr<FloatType>(),
        sigma[i].data_ptr<FloatType>());
    lnpsi += 0.5*torch::log(torch::amax(y[i]*sigma[i], 1));
  }
  return {lnpsi, spinStates};
}

template <typename FloatType>
at::Tensor pRNN<FloatType>::forward(const at::Tensor & spinStates)
{
  // stop tracking history on tensors that requires gradients
  torch::NoGradGuard no_grad;
  auto hiddens = GRULayer_(s0_, h0_);
  y_tmp_[0] = SoftmaxLayer_(LinearLayer_(hiddens));
  this->set_sigma_(spinStates[0].data_ptr<FloatType>(), sigma_tmp_[0].data_ptr<FloatType>());
  // lnpsi : \sum_i log(sqrt(y_i*sigma_i))
  auto lnpsi = 0.5*torch::log(torch::amax(y_tmp_[0]*sigma_tmp_[0], 1));
  for (int i=1; i<knInputs; ++i)
  {
    hiddens = GRULayer_(spinStates[i-1], hiddens);
    y_tmp_[i] = SoftmaxLayer_(LinearLayer_(hiddens));
    this->set_sigma_(spinStates[i].data_ptr<FloatType>(), sigma_tmp_[i].data_ptr<FloatType>());
    lnpsi += 0.5*torch::log(torch::amax(y_tmp_[i]*sigma_tmp_[i], 1));
  }
  return lnpsi;
}

template <typename FloatType>
void pRNN<FloatType>::sampling_states_(const FloatType * y, FloatType * spinState, FloatType * sigma)
{
  for (int k=0; k<knBatches; ++k)
  {
    const bool IsSpinUpState = randUniform_(randDev_[k]) < y[2*k+0];
    spinState[k] = (2*IsSpinUpState-1);
    // (1-isUpState)=0 (true, up state), (1-isUpState)=1 (false, down state)
    sigma[2*k+1-IsSpinUpState] = 1.0;
    sigma[2*k+IsSpinUpState] = 0.0;
  }
}

template <typename FloatType>
void pRNN<FloatType>::set_sigma_(const FloatType * spinState, FloatType * sigma) const
{
  for (int k=0; k<knBatches; ++k)
  {
    const bool IsSpinUpState = (spinState[k] > 0.0);
    sigma[2*k+1-IsSpinUpState] = 1.0;
    sigma[2*k+IsSpinUpState] = 0.0;
  }
}
