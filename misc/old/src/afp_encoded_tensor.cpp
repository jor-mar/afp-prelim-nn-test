#include "../include/afp_encoded_tensor.hpp"

#include "../include/afp_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace
{
    constexpr std::uint32_t exponent_mask = 0x7F800000U;
    constexpr std::uint32_t fraction_mask = 0x007FFFFFU;

    constexpr int fp32_exponent_bias = 127;
    constexpr int fp32_fraction_bits = 23;

    constexpr int block_size = 16;
    constexpr int half_block_size = 8;

    constexpr int maximum_offset = 7;

    constexpr std::uint8_t first_half_positive_bit = 0;
    constexpr std::uint8_t second_half_positive_bit = 1;

    std::uint32_t floatBits(float value)
    {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    std::uint64_t bitMask(int bit_count)
    {
        if (bit_count <= 0)
            return 0;

        return (std::uint64_t{1} << bit_count) - 1;
    }

    std::uint64_t roundShiftRight(std::uint64_t value, int shift)
    {
        if (shift <= 0)
            return value << -shift;

        if (shift >= 64)
            return 0;

        const std::uint64_t truncated = value >> shift;
        const std::uint64_t remainder = value & ((std::uint64_t{1} << shift) - 1);
        const std::uint64_t halfway = std::uint64_t{1} << (shift - 1);

        if (remainder > halfway)
            return truncated + 1;

        if (remainder == halfway && (truncated & 1))
            return truncated + 1;

        return truncated;
    }

    void validateConfig(const AFPConfig &config)
    {
        if (config.block_size != block_size)
            throw std::invalid_argument("AFP8 requires a block size of 16");

        if (config.exponent_bits != 8)
            throw std::invalid_argument("AFP8 requires an 8-bit shared exponent");

        if (config.characterization_bits != 8)
            throw std::invalid_argument("AFP8 requires an 8-bit characterization field");

        if (config.offset_bits != 3)
            throw std::invalid_argument("AFP8 requires a 3-bit offset");

        if (config.mantissa_bits != 5)
            throw std::invalid_argument("AFP8 requires a 5-bit mantissa");

        if (!config.enable_positive_fields)
            throw std::invalid_argument("AFP8 requires positive fields");

        if (config.enable_zero_fields)
            throw std::invalid_argument("AFP8 does not use zero fields");
    }

    int unbiasedExponent(float value)
    {
        const std::uint32_t bits = floatBits(value);
        const int exponent = static_cast<int>((bits & exponent_mask) >> fp32_fraction_bits);

        if (exponent == 0)
            return -126;

        return exponent - fp32_exponent_bias;
    }

    std::uint32_t significand(float value)
    {
        const std::uint32_t bits = floatBits(value);
        const std::uint32_t exponent = (bits & exponent_mask) >> fp32_fraction_bits;
        const std::uint32_t fraction = bits & fraction_mask;

        if (exponent == 0)
            return fraction;

        return (std::uint32_t{1} << fp32_fraction_bits) | fraction;
    }

    bool halfIsPositive(const std::vector<float> &input, std::size_t start, std::size_t end)
    {
        for (std::size_t i = start; i < end; ++i)
        {
            if (input[i] < 0.0f)
                return false;
        }

        return true;
    }

    int sharedExponent(const std::vector<float> &block)
    {
        bool found_nonzero = false;
        int maximum = -126;

        for (float value : block)
        {
            if (value == 0.0f)
                continue;

            const int exponent = unbiasedExponent(std::fabs(value));

            if (!found_nonzero || exponent > maximum)
            {
                maximum = exponent;
                found_nonzero = true;
            }
        }

        return maximum;
    }

    std::uint8_t encodeSharedExponent(int exponent)
    {
        if (exponent < -126 || exponent > 127)
            throw std::out_of_range("AFP shared exponent is outside FP32 normal range");

        return static_cast<std::uint8_t>(exponent + fp32_exponent_bias);
    }

    int decodeSharedExponent(std::uint8_t exponent)
    {
        if (exponent == 0)
            return -126;

        return static_cast<int>(exponent) - fp32_exponent_bias;
    }

    std::uint64_t encodeMantissa(float magnitude,
                                 int exponent,
                                 int offset,
                                 int mantissa_bits)
    {
        if (magnitude == 0.0f)
            return 0;

        const double scaled = std::ldexp(
            static_cast<double>(magnitude),
            -(exponent - offset));

        if (offset < maximum_offset)
        {
            const double fraction = scaled - 1.0;
            const double factor = static_cast<double>(std::uint64_t{1} << mantissa_bits);

            std::uint64_t mantissa = static_cast<std::uint64_t>(
                std::nearbyint(fraction * factor));

            const std::uint64_t maximum = bitMask(mantissa_bits);

            if (mantissa > maximum)
                mantissa = maximum;

            return mantissa;
        }

        const double factor = std::ldexp(
            1.0,
            exponent - maximum_offset - mantissa_bits);

        std::uint64_t mantissa = static_cast<std::uint64_t>(
            std::nearbyint(static_cast<double>(magnitude) / factor));

        const std::uint64_t maximum = bitMask(mantissa_bits);

        if (mantissa > maximum)
            mantissa = maximum;

        return mantissa;
    }

    float decodeValue(bool negative,
                      int exponent,
                      int offset,
                      std::uint64_t mantissa,
                      int mantissa_bits)
    {
        if (offset == maximum_offset && mantissa == 0)
            return 0.0f;

        double magnitude;

        if (offset < maximum_offset)
        {
            const double fraction =
                static_cast<double>(mantissa) /
                static_cast<double>(std::uint64_t{1} << mantissa_bits);

            magnitude = std::ldexp(1.0 + fraction, exponent - offset);
        }
        else
        {
            magnitude = std::ldexp(
                static_cast<double>(mantissa),
                exponent - maximum_offset - mantissa_bits);
        }

        const float result = static_cast<float>(magnitude);
        return negative ? -result : result;
    }

    std::uint64_t readBits(const BitStream &stream, std::size_t &bit_offset, int bit_count)
    {
        const std::uint64_t value = stream.readBits(
            bit_offset,
            static_cast<std::size_t>(bit_count));

        bit_offset += static_cast<std::size_t>(bit_count);
        return value;
    }

}

/*
AFPEncodedTensor AFPEncodedTensor::operator*(
    const AFPEncodedTensor &other
) const
{
    return AFPArithmetic::multiply(
        *this,
        other
    );
}

AFPEncodedTensor AFPEncodedTensor::operator+(
    const AFPEncodedTensor &other
) const
{
    return AFPArithmetic::add(
        *this,
        other
    );
}

AFPEncodedTensor AFPEncodedTensor::operator-(
    const AFPEncodedTensor &other
) const
{
    return AFPArithmetic::subtract(
        *this,
        other
    );
}

AFPEncodedTensor AFPEncodedTensor::operator-() const
{
    return AFPArithmetic::negate(*this);
}

AFPEncodedTensor AFPEncodedTensor::operator/(
    const AFPEncodedTensor &other
) const
{
    return AFPArithmetic::divide(
        *this,
        other
    );
}

bool AFPEncodedTensor::operator==(
    const AFPEncodedTensor &other
) const
{
    if (value_count_ != other.value_count_)
    {
        return false;
    }

    if (config_.mantissa_bits != other.config_.mantissa_bits)
    {
        return false;
    }

    if (config_.block_size != other.config_.block_size)
    {
        return false;
    }

    if (bits_.bitSize() != other.bits_.bitSize())
    {
        return false;
    }

    return bits_.data() == other.bits_.data();
}

bool AFPEncodedTensor::operator!=(
    const AFPEncodedTensor &other
) const
{
    return !(*this == other);
}
*/

AFPQuantizer::AFPQuantizer(AFPConfig config) : config_(config)
{
    validateConfig(config_);
}

AFPEncodedTensor AFPQuantizer::encode(const std::vector<float> &input) const
{
    AFPEncodedTensor encoded;

    encoded.config_ = config_;
    encoded.value_count_ = input.size();

    if (input.empty())
        return encoded;

    for (std::size_t block_start = 0;
         block_start < input.size();
         block_start += block_size)
    {
        encoded.block_offsets_.push_back(encoded.bits_.bitSize());

        std::vector<float> block(block_size, 0.0f);

        for (std::size_t i = 0; i < block_size && block_start + i < input.size(); ++i)
        {
            const float value = input[block_start + i];

            if (!std::isfinite(value))
                throw std::invalid_argument("AFP8 does not support NaN or infinity");

            block[i] = value;
        }

        const int exponent = sharedExponent(block);

        const bool first_half_positive =
            halfIsPositive(block, 0, half_block_size);

        const bool second_half_positive =
            halfIsPositive(block, half_block_size, block_size);

        std::uint8_t characterization = 0;

        if (first_half_positive)
            characterization |= std::uint8_t{1} << first_half_positive_bit;

        if (second_half_positive)
            characterization |= std::uint8_t{1} << second_half_positive_bit;

        encoded.bits_.writeBits(encodeSharedExponent(exponent), 8);
        encoded.bits_.writeBits(characterization, 8);

        for (std::size_t i = 0; i < block_size; ++i)
        {
            const float value = block[i];
            const bool positive_half =
                i < half_block_size ? first_half_positive : second_half_positive;

            const int effective_mantissa_bits =
                config_.mantissa_bits + (positive_half ? 1 : 0);

            const bool negative = std::signbit(value) && value != 0.0f;

            int offset = maximum_offset;
            std::uint64_t mantissa = 0;

            if (value != 0.0f)
            {
                const int value_exponent = unbiasedExponent(std::fabs(value));
                const int true_offset = exponent - value_exponent;

                // offset = std::clamp(true_offset, 0, maximum_offset);
                if (true_offset < 0) {
                    offset = 0;
                }
                else if (true_offset > maximum_offset) {
                    offset = maximum_offset;
                }
                else {
                    offset = true_offset;
                }

                mantissa = encodeMantissa(
                    std::fabs(value),
                    exponent,
                    offset,
                    effective_mantissa_bits);
            }

            if (positive_half)
            {
                const std::uint64_t extra_bit =
                    mantissa >> config_.mantissa_bits;

                const std::uint64_t stored_mantissa =
                    mantissa & bitMask(config_.mantissa_bits);

                encoded.bits_.writeBits(extra_bit, 1);
                encoded.bits_.writeBits(offset, 3);
                encoded.bits_.writeBits(stored_mantissa, 5);
            }
            else
            {
                encoded.bits_.writeBits(negative ? 1 : 0, 1);
                encoded.bits_.writeBits(offset, 3);
                encoded.bits_.writeBits(mantissa, 5);
            }
        }
    }

    return encoded;
}

std::vector<float> AFPQuantizer::decode(const AFPEncodedTensor &encoded) const
{
    validateConfig(encoded.config_);

    std::vector<float> output;
    output.reserve(encoded.value_count_);

    if (encoded.value_count_ == 0)
        return output;

    std::size_t bit_offset = 0;
    std::size_t values_decoded = 0;

    while (values_decoded < encoded.value_count_)
    {
        const std::uint8_t stored_exponent =
            static_cast<std::uint8_t>(readBits(encoded.bits_, bit_offset, 8));

        const int exponent = decodeSharedExponent(stored_exponent);

        const std::uint8_t characterization =
            static_cast<std::uint8_t>(readBits(encoded.bits_, bit_offset, 8));

        const bool first_half_positive =
            (characterization & (std::uint8_t{1} << first_half_positive_bit)) != 0;

        const bool second_half_positive =
            (characterization & (std::uint8_t{1} << second_half_positive_bit)) != 0;

        for (std::size_t i = 0; i < block_size; ++i)
        {
            const bool positive_half =
                i < half_block_size ? first_half_positive : second_half_positive;

            const std::uint64_t first_field =
                readBits(encoded.bits_, bit_offset, 1);

            const int offset =
                static_cast<int>(readBits(encoded.bits_, bit_offset, 3));

            std::uint64_t mantissa =
                readBits(encoded.bits_, bit_offset, 5);

            bool negative = false;
            int effective_mantissa_bits = config_.mantissa_bits;

            if (positive_half)
            {
                mantissa |= first_field << config_.mantissa_bits;
                effective_mantissa_bits++;
            }
            else
            {
                negative = first_field != 0;
            }

            if (values_decoded < encoded.value_count_)
            {
                output.push_back(
                    decodeValue(
                        negative,
                        exponent,
                        offset,
                        mantissa,
                        effective_mantissa_bits));

                ++values_decoded;
            }
        }
    }

    return output;
}

AFPEncodedTensor AFPQuantizer::load(
    const std::vector<std::uint8_t> &data,
    std::size_t bit_size,
    std::size_t value_count,
    const AFPConfig &config,
    const std::vector<std::size_t> &block_offsets) const
{
    AFPEncodedTensor result;

    result.config_ = config;
    result.value_count_ = value_count;
    result.block_offsets_ = block_offsets;

    result.bits_.loadData(
        data,
        bit_size
    );

    return result;
}

const AFPConfig &AFPQuantizer::getConfig() const
{
    return config_;
}