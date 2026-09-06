#pragma once

#include "encoded_tensor.hpp"

#include <vector>

struct INT8Config
{
  bool symmetric = true;
};

class INT8EncodedTensor : public EncodedTensor
{
public:
  std::string formatName() const override
  {
    return "INT8";
  }

  float scale() const
  {
    return scale_;
  }

private:
  friend class INT8Quantizer;
  float scale_ = 1.0f;
};

class INT8Quantizer
{
public:
  explicit INT8Quantizer(
    INT8Config config = {}
  );

  INT8EncodedTensor encode(
    const std::vector<float>& input
  ) const;


  std::vector<float> decode(
    const INT8EncodedTensor& encoded
  ) const;

  const INT8Config& getConfig() const;

private:
  INT8Config config_;
};