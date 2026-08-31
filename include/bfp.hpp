#pragma once

#include "encoded_tensor.hpp"

#include <cstddef>
#include <vector>

struct BFPConfig
{
    std::size_t block_size = 16;
    int mantissa_bits = 8;
    int exponent_bits = 8;
};

class BFPEncodedTensor : public EncodedTensor
{
public:
    std::string formatName() const override
    {
        return "BFP";
    }

    std::size_t blockSize() const
    {
        return config_.block_size;
    }


    int mantissaBits() const
    {
        return config_.mantissa_bits;
    }


    int exponentBits() const
    {
        return config_.exponent_bits;
    }

private:
    friend class BFPQuantizer;
    BFPConfig config_;
    std::vector<std::size_t> block_offsets_;
};

class BFPQuantizer
{
public:
    explicit BFPQuantizer(
    BFPConfig config = {}
    );

    BFPEncodedTensor encode(
        const std::vector<float>& input
    ) const;

    std::vector<float> decode(
        const BFPEncodedTensor& encoded
    ) const;

    const BFPConfig& getConfig() const;

private:
    BFPConfig config_;
};