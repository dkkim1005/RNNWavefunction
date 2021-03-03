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
  kgpuBlockSize(CHECK_BLOCK_SIZE(1+(nBatches-1)/NUM_THREADS_PER_BLOCK)),
  GRULayer_(register_module("GRU-1 Layer", torch::nn::GRUCell(1, nHiddens))),
  LinearLayer_(register_module("Softmax Layer", torch::nn::Linear(nHiddens, 2))),
  SoftmaxLayer_(1),
  s0_(torch::zeros({nBatches, 1},
    torch::TensorOptions()
      .dtype(TorchPrec<FloatType>::dtype)
      .device(torch::kCUDA)
    )),
  h0_(torch::zeros({nBatches, nHiddens},
    torch::TensorOptions()
      .dtype(TorchPrec<FloatType>::dtype)
      .device(torch::kCUDA)
    )),
  y_tmp_(torch::zeros({nInputs, nBatches, 2}, 
    torch::TensorOptions()
      .dtype(TorchPrec<FloatType>::dtype)
      .device(torch::kCUDA)
    )),
  sigma_tmp_(torch::zeros({nInputs, nBatches, 2}, 
    torch::TensorOptions()
      .dtype(TorchPrec<FloatType>::dtype)
      .device(torch::kCUDA)
    )),
  randDev_(seedNumber, seedDistance, nBatches),
  randVal_dev_(nBatches, 0.0),
  options_(
    torch::TensorOptions()
      .dtype(TorchPrec<FloatType>::dtype)
      .device(torch::kCUDA)
    )
{
  this->to(TorchPrec<FloatType>::dtype);
  this->to(torch::kCUDA);
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
  FloatType * randVal_ptr_dev = PTR_FROM_THRUST(randVal_dev_.data());
  // get random numbers
  randDev_.get_uniformDist(randVal_ptr_dev);
  gpu_kernel::sampling_states<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knBatches,
    y[0].data_ptr<FloatType>(),
    randVal_ptr_dev,
    spinStates[0].data_ptr<FloatType>(),
    sigma[0].data_ptr<FloatType>());
  // lnpsi : \sum_i log(sqrt(y_i*sigma_i))
  auto lnpsi = torch::zeros({knBatches}, options_);
  lnpsi += 0.5*torch::log(torch::amax(y[0]*sigma[0], 1));
  for (int i=1; i<knInputs; ++i)
  {
    hiddens = GRULayer_(spinStates[i-1], hiddens);
    y[i] = SoftmaxLayer_(LinearLayer_(hiddens));
    // get random numbers
    randDev_.get_uniformDist(randVal_ptr_dev);
    gpu_kernel::sampling_states<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knBatches,
      y[i].data_ptr<FloatType>(),
      randVal_ptr_dev,
      spinStates[i].data_ptr<FloatType>(),
      sigma[i].data_ptr<FloatType>());
    lnpsi += 0.5*torch::log(torch::amax(y[i]*sigma[i], 1));
  }
  return {lnpsi, spinStates};
}

template <typename FloatType>
at::Tensor pRNN<FloatType>::forward_no_grad(const at::Tensor & spinStates)
{
  // stop tracking history on tensors that requires gradients
  torch::NoGradGuard no_grad;
  auto hiddens = GRULayer_(s0_, h0_);
  y_tmp_[0] = SoftmaxLayer_(LinearLayer_(hiddens));
  gpu_kernel::set_sigma<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knBatches,
    spinStates[0].data_ptr<FloatType>(), sigma_tmp_[0].data_ptr<FloatType>());
  // lnpsi : \sum_i log(sqrt(y_i*sigma_i))
  auto lnpsi = 0.5*torch::log(torch::amax(y_tmp_[0]*sigma_tmp_[0], 1));
  for (int i=1; i<knInputs; ++i)
  {
    hiddens = GRULayer_(spinStates[i-1], hiddens);
    y_tmp_[i] = SoftmaxLayer_(LinearLayer_(hiddens));
    gpu_kernel::set_sigma<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knBatches,
      spinStates[i].data_ptr<FloatType>(), sigma_tmp_[i].data_ptr<FloatType>());
    lnpsi += 0.5*torch::log(torch::amax(y_tmp_[i]*sigma_tmp_[i], 1));
  }
  return lnpsi;
}

template <typename FloatType>
at::Tensor pRNN<FloatType>::forward_with_grad(const at::Tensor & spinStates)
{
  auto y = torch::zeros({knInputs, knBatches, 2}, options_);
  auto sigma = torch::zeros({knInputs, knBatches, 2}, options_);
  auto hiddens = GRULayer_(s0_, h0_);
  y[0] = SoftmaxLayer_(LinearLayer_(hiddens));
  gpu_kernel::set_sigma<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knBatches,
    spinStates[0].data_ptr<FloatType>(), sigma[0].data_ptr<FloatType>());
  // lnpsi : \sum_i log(sqrt(y_i*sigma_i))
  auto lnpsi = torch::zeros({knBatches}, options_);
  lnpsi += 0.5*torch::log(torch::amax(y[0]*sigma[0], 1));
  for (int i=1; i<knInputs; ++i)
  {
    hiddens = GRULayer_(spinStates[i-1], hiddens);
    y[i] = SoftmaxLayer_(LinearLayer_(hiddens));
    gpu_kernel::set_sigma<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knBatches,
      spinStates[i].data_ptr<FloatType>(), sigma[i].data_ptr<FloatType>());
    lnpsi += 0.5*torch::log(torch::amax(y[i]*sigma[i], 1));
  }
  return lnpsi;
}


namespace gpu_kernel
{
template <typename FloatType>
__global__ void sampling_states(const int nBatches, const FloatType * y, const FloatType * randVal,
  FloatType * spinState, FloatType * sigma)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nBatches)
  {
    const bool IsSpinUpState = randVal[idx] < y[2*idx+0];
    spinState[idx] = (2*IsSpinUpState-1);
    // (1-isUpState)=0 (true, up state), (1-isUpState)=1 (false, down state)
    sigma[2*idx+1-IsSpinUpState] = 1.0;
    sigma[2*idx+IsSpinUpState] = 0.0;
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void set_sigma(const int nBatches, const FloatType * spinState, FloatType * sigma)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nBatches)
  {
    const bool IsSpinUpState = (spinState[idx] > 0.0);
    sigma[2*idx+1-IsSpinUpState] = 1.0;
    sigma[2*idx+IsSpinUpState] = 0.0;
    idx += nstep;
  }
}
} // end namespace gpu_kernel
