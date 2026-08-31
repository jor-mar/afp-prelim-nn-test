#pragma once

#include "encoded_tensor.hpp"

#include <cstddef>
#include <vector>

struct AFPConfig
{
    std::size_t block_size = 16;
    int offset_bits = 3;
    int mantissa_bits = 5;
    int exponent_bits = 8;
    bool enable_positive_block_optimization = false; // false until it's known block is fully positive
    bool enable_zero_bit_optimization = false;
};

class AFPEncodedTensor : public EncodedTensor
{
public:
    std::string formatName() const override
    {
        return "AFP";
    }

    std::size_t blockSize() const
    {
        return config_.block_size;
    }

    int offsetBits() const
    {
        return config_.offset_bits;
    }

    int mantissaBits() const
    {
        return config_.mantissa_bits;
    }

    int exponentBits() const
    {
        return config_.exponent_bits;
    }

    bool positiveBlockOptimizationEnabled() const
    {
        return config_.enable_positive_block_optimization;
    }

    bool zeroBitOptimizationEnabled() const
    {
        return config_.enable_zero_bit_optimization;
    }

private:
    friend class AFPQuantizer;
    AFPConfig config_;
    std::vector<std::size_t> block_offsets_;
};

class AFPQuantizer
{
public:
    explicit AFPQuantizer(
        AFPConfig config = {}
    );

    AFPEncodedTensor encode(
        const std::vector<float>& input
    ) const;

    std::vector<float> decode(
        const AFPEncodedTensor& encoded
    ) const;

    const AFPConfig& getConfig() const;

private:
    AFPConfig config_;
};
