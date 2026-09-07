#pragma once

#include "encoded_tensor.hpp"
#include "afp_value.hpp"

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

class AFPArithmetic;

class AFPEncodedTensor : public EncodedTensor
{
public:
    std::string formatName() const override { return "AFP8"; }
    
    std::size_t blockSize() const { return config_.block_size; }
    int exponentBits() const { return config_.exponent_bits; }
    int characterizationBits() const { return config_.characterization_bits; }
    int offsetBits() const { return config_.offset_bits; }
    int mantissaBits() const { return config_.mantissa_bits; }
    bool positiveFieldsEnabled() const { return config_.enable_positive_fields; }
    bool zeroFieldsEnabled() const { return config_.enable_zero_fields; }
    
    std::size_t blockCount() const { return block_offsets_.size(); }
    const std::vector<std::size_t>& blockOffsets() const { return block_offsets_; }
    const AFPConfig& config() const { return config_; }

private:
    friend class AFPQuantizer;
    friend class AFPArithmetic;

    AFPConfig config_;
    std::vector<std::size_t> block_offsets_;
};

class AFPQuantizer
{
public:
    explicit AFPQuantizer(AFPConfig config = {});

    AFPEncodedTensor encode(const std::vector<float>& input) const;
    std::vector<float> decode(const AFPEncodedTensor& encoded) const;
    
    AFPEncodedTensor load(
        const std::vector<std::uint8_t>& data,
        std::size_t bit_size,
        std::size_t value_count,
        const AFPConfig& config,
        const std::vector<std::size_t>& block_offsets) const;

    const AFPConfig& getConfig() const { return config_; }

private:
    AFPConfig config_;
    
    // Helper methods for encoding/decoding
    static int unbiasedExponent(float value);
    static uint32_t significand(float value);
    static bool halfIsPositive(const std::vector<float>& input, std::size_t start, std::size_t end);
    static int sharedExponent(const std::vector<float>& block);
    static uint8_t encodeSharedExponent(int exponent);
    static int decodeSharedExponent(uint8_t exponent);
    static uint64_t encodeMantissa(float magnitude, int exponent, int offset, int mantissa_bits);
    static float decodeValue(bool negative, int exponent, int offset, uint64_t mantissa, int mantissa_bits);
    static void validateConfig(const AFPConfig& config);
    
    // Constants
    static constexpr int block_size = 16;
    static constexpr int half_block_size = 8;
    static constexpr int maximum_offset = 7;
    static constexpr int fp32_exponent_bias = 127;
    static constexpr int fp32_fraction_bits = 23;
};