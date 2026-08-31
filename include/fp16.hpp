#pragma once

#include "encoded_tensor.hpp"

#include <cstdint>
#include <vector>

class FP16EncodedTensor : public EncodedTensor
{
public:
    std::string formatName() const override
    {
        return "FP16";
    }
};

class FP16Quantizer
{
public:
    FP16EncodedTensor encode(
        const std::vector<float>& input
    ) const;

    std::vector<float> decode(
        const FP16EncodedTensor& encoded
    ) const;
};
