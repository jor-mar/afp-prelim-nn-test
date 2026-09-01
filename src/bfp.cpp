#include "../include/bfp.hpp"

#include "../include/quantization_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace
{
    std::int64_t signExtend(std::uint64_t value, int bits)
    {
        if (bits <= 0 || bits >= 64)
        {
            return static_cast<std::int64_t>(value);
        }

        const std::uint64_t sign_bit = std::uint64_t{1} << (bits - 1);

        if ((value & sign_bit) == 0)
        {
            return static_cast<std::int64_t>(value);
        }

        const std::uint64_t extension_mask = ~((std::uint64_t{1} << bits) - 1);

        return static_cast<std::int64_t>(value | extension_mask);
    }

    std::uint64_t encodeSigned(std::int64_t value, int bits)
    {
        if (bits <= 0 || bits > 63)
        {
            throw std::invalid_argument("Invalid signed bit width");
        }

        const std::int64_t minimum = -(std::int64_t{1} << (bits - 1));

        const std::int64_t maximum = (std::int64_t{1} << (bits - 1)) - 1;

        if (value < minimum || value > maximum)
        {
            throw std::out_of_range("Exponent cannot fit in configured bit width");
        }

        const std::uint64_t mask = (std::uint64_t{1} << bits) - 1;

        return static_cast<std::uint64_t>(value) & mask;
    }

}

BFPQuantizer::BFPQuantizer(BFPConfig config) : config_(config)
{
    if (config_.block_size == 0)
    {
        throw std::invalid_argument("block_size must be greater than zero");
    }

    if (config_.mantissa_bits <= 0 || config_.mantissa_bits >= 63)
    {
        throw std::invalid_argument("mantissa_bits must be between 1 and 62");
    }

    if (config_.exponent_bits <= 1 || config_.exponent_bits > 63)
    {
        throw std::invalid_argument("Invalid exponent_bits");
    }
}

BFPEncodedTensor BFPQuantizer::encode(const std::vector<float> &input) const
{
    BFPEncodedTensor encoded;

    encoded.config_ = config_;
    encoded.value_count_ = input.size();

    const std::uint64_t max_mantissa = (std::uint64_t{1} << config_.mantissa_bits) - 1;

    for (std::size_t block_start = 0; block_start < input.size(); block_start += config_.block_size)
    {
        const std::size_t block_end = std::min(block_start + config_.block_size, input.size());

        encoded.block_offsets_.push_back(encoded.bits_.bitSize());

        float max_abs = 0.0f;

        for (std::size_t i = block_start; i < block_end; ++i)
        {
            max_abs = std::max(max_abs, std::abs(input[i]));
        }

        int shared_exponent = 0;

        if (max_abs != 0.0f)
        {
            shared_exponent = quantization_utils::getExponent(max_abs);
        }

        encoded.bits_.writeBits(encodeSigned(shared_exponent, config_.exponent_bits), config_.exponent_bits);

        const float scale = std::ldexp(1.0f, shared_exponent) / static_cast<float>(max_mantissa);

        for (std::size_t i = block_start; i < block_end; ++i)
        {
            const float value = input[i];

            const std::uint64_t sign = value < 0.0f ? 1 : 0;

            std::uint64_t mantissa = 0;

            if (value != 0.0f)
            {
                mantissa = static_cast<std::uint64_t>(std::llround(std::abs(value) / scale));

                mantissa = std::min(mantissa, max_mantissa);
            }

            encoded.bits_.writeBits(sign, 1);

            encoded.bits_.writeBits(mantissa, config_.mantissa_bits);
        }
    }

    return encoded;
}

std::vector<float> BFPQuantizer::decode(const BFPEncodedTensor &encoded) const
{
    std::vector<float> output;

    output.reserve(encoded.value_count_);

    const BFPConfig &config = encoded.config_;

    const std::uint64_t max_mantissa = (std::uint64_t{1} << config.mantissa_bits) - 1;

    for (std::size_t block_index = 0; block_index < encoded.block_offsets_.size(); ++block_index)
    {
        std::size_t bit_offset = encoded.block_offsets_[block_index];

        const std::uint64_t raw_exponent = encoded.bits_.readBits(bit_offset, config.exponent_bits);

        bit_offset += config.exponent_bits;

        const int shared_exponent = static_cast<int>(signExtend(raw_exponent, config.exponent_bits));

        const float scale = std::ldexp(1.0f, shared_exponent) / static_cast<float>(max_mantissa);

        const std::size_t values_in_block = std::min(config.block_size, encoded.value_count_ - output.size());

        for (std::size_t i = 0; i < values_in_block; ++i)
        {
            const bool negative = encoded.bits_.readBits(bit_offset, 1) != 0;

            bit_offset += 1;

            const std::uint64_t mantissa = encoded.bits_.readBits(bit_offset, config.mantissa_bits);

            bit_offset += config.mantissa_bits;

            float value = static_cast<float>(mantissa) * scale;

            if (negative)
            {
                value = -value;
            }

            output.push_back(value);
        }
    }

    return output;
}

const BFPConfig &BFPQuantizer::getConfig() const
{
    return config_;
}