#pragma once

#include "encoded_tensor.hpp"

#include <vector>

class BF16EncodedTensor : public EncodedTensor
{
public:
  std::string formatName() const override
  {
    return "BF16";
  }
};

class BF16Quantizer
{
public:
  BF16EncodedTensor encode(
    const std::vector<float>& input
  ) const;

  std::vector<float> decode(
    const BF16EncodedTensor& encoded
  ) const;
};