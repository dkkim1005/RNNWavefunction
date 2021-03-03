#include <iostream>
#include <fstream>
#include "../include/rnnwavefunction.hpp"
#include "../include/hamiltonians.hpp"
#include "../include/argparse.hpp"

template <typename FloatType>
std::string remove_zeros_in_str(const FloatType val);

template <typename TorchNetworkType>
void load_parameters(const std::string & filepath, TorchNetworkType & TorchNet);

template <typename TorchNetworkType>
void save_parameters(const std::string & filepath, const TorchNetworkType & TorchNet);

int main(int argc, char * argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("L", "# of lattice sites"));
  options.push_back(pair_t("nh", "# of hidden cells"));
  options.push_back(pair_t("nb", "# of batches"));
  options.push_back(pair_t("niter", "# of iterations"));
  options.push_back(pair_t("h", "transverse-field strength"));
  options.push_back(pair_t("J", "coupling constant"));
  options.push_back(pair_t("lr", "learning_rate"));
  options.push_back(pair_t("ver", "version"));
  options.push_back(pair_t("path", "directory to load and save files"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  // env; default value
  defaults.push_back(pair_t("J", "-1.0"));
  defaults.push_back(pair_t("lr", "5e-3"));
  defaults.push_back(pair_t("path", "."));
  defaults.push_back(pair_t("seed", "0"));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);

  parser.print(std::cout);

  const int nInputs = parser.find<int>("L"),
    nHiddens = parser.find<int>("nh"),
    nBatches = parser.find<int>("nb"),
    niter = parser.find<int>("niter");
  const float J = parser.find<float>("J"),
    h = parser.find<float>("h"),
    learning_rate = parser.find<float>("lr");
  const unsigned long long seed = parser.find<unsigned long long>("seed");
  const std::string filepath = parser.find<>("path")
    + "/pRNN-TFICH-L" + parser.find<>("L")
    + "NH" + parser.find<>("nh")
    + "H" + remove_zeros_in_str(h)
    + "V" + parser.find<>("ver") + ".pt";
  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(niter)*
    static_cast<unsigned long>(nInputs)*
    static_cast<unsigned long>(nBatches);

  pRNN<float> ansatz(nInputs, nHiddens, nBatches, seed, nBlocks);
  TFIChain<float> hamil(J, h);
  torch::optim::Adam optimizer(ansatz.parameters(), learning_rate);

  try
  {
    load_parameters(filepath, ansatz);
  }
  catch (const std::exception & e)
  {
    std::cout << "# There is no file name: " << filepath << std::endl;
  }

  std::cout << "# of loop\t" << "<H>" << std::endl << std::setprecision(7);
  for (int n=0; n<niter; ++n)
  {
    optimizer.zero_grad();
    auto res = hamil.construct_loss(ansatz);
    auto & loss = res[0];
    auto & htilda = res[1];
    loss.backward();
    optimizer.step();
    const float E = htilda.mean().item<float>();
    std::cout << std::setw(5) << (n+1) << std::setw(16)
      << E << std::setw(16)
      << std::sqrt((htilda-E).pow(2).mean().item<float>())/std::abs(E)
      << std::endl;
    if (n%100 == 99)
      save_parameters(filepath, ansatz);
  }

  save_parameters(filepath, ansatz);

  return 0;
}

template <typename FloatType>
std::string remove_zeros_in_str(const FloatType val)
{
  std::string tmp = std::to_string(val);
  tmp.erase(tmp.find_last_not_of('0') + 1, std::string::npos);
  tmp.erase(tmp.find_last_not_of('.') + 1, std::string::npos);
  return tmp;
}

template <typename TorchNetworkType>
void load_parameters(const std::string & filepath, TorchNetworkType & TorchNet)
{
  torch::serialize::InputArchive input_archive;
  input_archive.load_from(filepath);
  TorchNet.load(input_archive);
}

template <typename TorchNetworkType>
void save_parameters(const std::string & filepath, const TorchNetworkType & TorchNet)
{
  torch::serialize::OutputArchive output_archive;
  TorchNet.save(output_archive);
  output_archive.save_to(filepath);
}
