#pragma once

#include "encoded_tensor.hpp"

#include <cstddef>
#include <string>
#include <vector>

struct AFPConfig
{
    std::size_t block_size = 16;

    int exponent_bits = 8;
    int characterization_bits = 8;

    int offset_bits = 3;
    int mantissa_bits = 5;

    bool enable_positive_fields = true;
    bool enable_zero_fields = false;
};

class AFPEncodedTensor : public EncodedTensor
{
public:
    std::string formatName() const override
    {
        return "AFP8";
    }

    std::size_t blockSize() const
    {
        return config_.block_size;
    }

    int exponentBits() const
    {
        return config_.exponent_bits;
    }

    int characterizationBits() const
    {
        return config_.characterization_bits;
    }

    int offsetBits() const
    {
        return config_.offset_bits;
    }

    int mantissaBits() const
    {
        return config_.mantissa_bits;
    }

    bool positiveFieldsEnabled() const
    {
        return config_.enable_positive_fields;
    }

    bool zeroFieldsEnabled() const
    {
        return config_.enable_zero_fields;
    }

    std::size_t blockCount() const
    {
        return block_offsets_.size();
    }

private:
    friend class AFPQuantizer;

    AFPConfig config_;
    std::vector<std::size_t> block_offsets_;
};

class AFPQuantizer
{
public:
    explicit AFPQuantizer(AFPConfig config = {});

    AFPEncodedTensor encode(const std::vector<float> &input) const;

    std::vector<float> decode(const AFPEncodedTensor &encoded) const;

    const AFPConfig &getConfig() const;

private:
    AFPConfig config_;
};
