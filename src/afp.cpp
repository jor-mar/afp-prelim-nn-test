#include "../include/afp.hpp"

#include "quantization_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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
            throw std::invalid_argument("Invalid exponent bit width");
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

AFPQuantizer::AFPQuantizer(AFPConfig config) : config_(config)
{
    if (config_.block_size == 0)
    {
        throw std::invalid_argument("block_size must be greater than zero");
    }

    if (config_.offset_bits <= 0 || config_.offset_bits >= 63)
    {
        throw std::invalid_argument("offset_bits must be between 1 and 62");
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

AFPEncodedTensor AFPQuantizer::encode(const std::vector<float> &input) const
{
    AFPEncodedTensor encoded;

    encoded.config_ = config_;

    encoded.value_count_ = input.size();

    const std::uint64_t max_offset = (std::uint64_t{1} << config_.offset_bits) - 1;

    const std::uint64_t max_mantissa = (std::uint64_t{1} << config_.mantissa_bits) - 1;

    for (std::size_t block_start = 0; block_start < input.size(); block_start += config_.block_size)
    {
        const std::size_t block_end = std::min(block_start + config_.block_size, input.size());
        encoded.block_offsets_.push_back(encoded.bits_.bitSize());

        int shared_exponent = 0;
        bool has_nonzero = false;
        for (std::size_t i = block_start; i < block_end; ++i)
        {
            const float value = input[i];

            if (value == 0.0f)
            {
                continue;
            }

            const int exponent = quantization_utils::getExponent(value);

            if (!has_nonzero || exponent > shared_exponent)
            {
                shared_exponent = exponent;
            }

            has_nonzero = true;
        }

        encoded.bits_.writeBits(encodeSigned(shared_exponent, config_.exponent_bits), config_.exponent_bits);


        for (std::size_t i = block_start; i < block_end; ++i)
        {
            const float value = input[i];

            if (value == 0.0f)
            {
                encoded.bits_.writeBits(0, 1);
                encoded.bits_.writeBits(max_offset, config_.offset_bits);
                encoded.bits_.writeBits(0, config_.mantissa_bits);
                continue;
            }

            const std::uint64_t sign = value < 0.0f ? 1 : 0;
            const int value_exponent = quantization_utils::getExponent(value);
            int offset = shared_exponent - value_exponent;
            offset = std::max(offset, 0);
            offset = std::min(offset, static_cast<int>(max_offset));

            const int effective_exponent = shared_exponent - offset;

            const float scale = std::ldexp(1.0f, effective_exponent) / static_cast<float>(max_mantissa);

            std::uint64_t mantissa = static_cast<std::uint64_t>(std::llround(std::abs(value) / scale));
            mantissa = std::min(mantissa, max_mantissa);
            encoded.bits_.writeBits(sign,1);

            encoded.bits_.writeBits(static_cast<std::uint64_t>(offset), config_.offset_bits);

            encoded.bits_.writeBits(mantissa, config_.mantissa_bits);
        }
    }
    return encoded;
}

std::vector<float> AFPQuantizer::decode(const AFPEncodedTensor &encoded) const
{
    std::vector<float> output;

    output.reserve(encoded.value_count_);

    const AFPConfig &config = encoded.config_;

    const std::uint64_t max_offset = (std::uint64_t{1} << config.offset_bits) - 1;

    const std::uint64_t max_mantissa = (std::uint64_t{1} << config.mantissa_bits) - 1;

    for (std::size_t block_index = 0; block_index < encoded.block_offsets_.size(); ++block_index)
    {
        std::size_t bit_offset = encoded.block_offsets_[block_index];

        const std::uint64_t raw_exponent = encoded.bits_.readBits(bit_offset, config.exponent_bits);

        bit_offset += config.exponent_bits;

        const int shared_exponent = static_cast<int>(signExtend(raw_exponent, config.exponent_bits));

        const std::size_t values_in_block = std::min(config.block_size, encoded.value_count_ - output.size());

        for (std::size_t i = 0; i < values_in_block; ++i)
        {
            const bool negative = encoded.bits_.readBits(bit_offset, 1) != 0;

            bit_offset += 1;

            const std::uint64_t offset = encoded.bits_.readBits(bit_offset, config.offset_bits);

            bit_offset += config.offset_bits;

            const std::uint64_t mantissa = encoded.bits_.readBits(bit_offset, config.mantissa_bits);

            bit_offset += config.mantissa_bits;

            if (mantissa == 0)
            {
                output.push_back(0.0f);

                continue;
            }

            if (offset > max_offset)
            {
                throw std::runtime_error("Invalid AFP exponent offset");
            }

            const int effective_exponent = shared_exponent - static_cast<int>(offset);

            const float scale = std::ldexp(1.0f, effective_exponent) / static_cast<float>(max_mantissa);

            float value =static_cast<float>(mantissa) * scale;

            if (negative)
            {
                value = -value;
            }

            output.push_back(value);
        }
    }

    return output;
}

const AFPConfig &AFPQuantizer::getConfig() const
{
    return config_;
}