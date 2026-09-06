#pragma once

#include "encoded_tensor.hpp"

#include <vector>

class FP16EncodedTensor : public EncodedTensor
{
public:
    std::string formatName() const override
    {
        return "FP16";
    }
private:
    friend class FP16Quantizer;
};

class FP16Quantizer
{
public:
    FP16EncodedTensor encode(const std::vector<float>& input) const;
    std::vector<float> decode(const FP16EncodedTensor& encoded) const;
};
