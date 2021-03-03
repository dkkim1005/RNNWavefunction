#pragma once

template <typename FloatType>
struct TorchPrec;

template <>
struct TorchPrec<float> {
  static constexpr auto dtype = torch::kFloat32;
};

template <>
struct TorchPrec<double> {
  static constexpr auto dtype = torch::kFloat64;
};
