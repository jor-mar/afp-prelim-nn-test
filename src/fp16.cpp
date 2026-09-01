#include "../include/fp16.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
    std::uint32_t floatToBits(float value)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(float));
        return bits;
    }

    float bitsToFloat(std::uint32_t bits)
    {
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(float));
        return value;
    }

    std::uint16_t floatToHalf(
        float value)
    {
        const std::uint32_t bits = floatToBits(value);

        const std::uint32_t sign = (bits >> 16) & 0x8000U;

        const std::uint32_t exponent = (bits >> 23) & 0xFFU;

        const std::uint32_t mantissa = bits & 0x7FFFFFU;

        if (exponent == 0xFFU)
        {
            if (mantissa == 0)
            {
                return static_cast<std::uint16_t>(sign | 0x7C00U);
            }

            std::uint16_t half_mantissa = static_cast<std::uint16_t>(mantissa >> 13);

            if (half_mantissa == 0)
            {
                half_mantissa = 1;
            }

            return static_cast<std::uint16_t>(sign | 0x7C00U | half_mantissa);
        }

        const int half_exponent = static_cast<int>(exponent) - 127 + 15;

        if (half_exponent >= 31)
        {
            return static_cast<std::uint16_t>(sign | 0x7C00U);
        }

        if (half_exponent <= 0)
        {
            if (half_exponent < -10)
            {
                return static_cast<std::uint16_t>(sign);
            }

            std::uint32_t subnormal_mantissa = mantissa | 0x800000U;

            const int shift = 14 - half_exponent;

            std::uint32_t half_mantissa = subnormal_mantissa >> shift;

            const std::uint32_t round_bit = std::uint32_t{1} << (shift - 1);

            if ((subnormal_mantissa & round_bit) != 0)
            {
                ++half_mantissa;
            }

            return static_cast<std::uint16_t>(sign | half_mantissa);
        }

        std::uint32_t half_mantissa = mantissa >> 13;

        const std::uint32_t remainder = mantissa & 0x1FFFU;

        if (remainder > 0x1000U || (remainder == 0x1000U && (half_mantissa & 1U)))
        {
            ++half_mantissa;
            if (half_mantissa == 0x400U)
            {
                half_mantissa = 0;

                const int rounded_exponent = half_exponent + 1;

                if (rounded_exponent >= 31)
                {
                    return static_cast<std::uint16_t>(sign | 0x7C00U);
                }

                return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(rounded_exponent) << 10));
            }
        }

        return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(half_exponent) << 10) | half_mantissa);
    }

    float halfToFloat(std::uint16_t half)
    {
        const std::uint32_t sign = (static_cast<std::uint32_t>(half) & 0x8000U) << 16;

        const std::uint32_t exponent = (static_cast<std::uint32_t>(half) >> 10) & 0x1FU;

        std::uint32_t mantissa = static_cast<std::uint32_t>(half) & 0x3FFU;

        std::uint32_t float_bits = 0;

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                float_bits = sign;
            }
            else
            {
                int normalized_exponent = -14;

                while ((mantissa & 0x400U) == 0)
                {
                    mantissa <<= 1;
                    --normalized_exponent;
                }

                mantissa &= 0x3FFU;

                const std::uint32_t float_exponent = static_cast<std::uint32_t>(normalized_exponent + 127);

                float_bits = sign | (float_exponent << 23) | (mantissa << 13);
            }
        }

        else if (exponent == 0x1FU)
        {
            float_bits = sign | 0x7F800000U | (mantissa << 13);
        }

        else
        {
            const std::uint32_t float_exponent = exponent - 15 + 127;

            float_bits = sign | (float_exponent << 23) | (mantissa << 13);
        }

        return bitsToFloat(float_bits);
    }

}

FP16EncodedTensor FP16Quantizer::encode(const std::vector<float> &input) const
{
    FP16EncodedTensor encoded;

    encoded.value_count_ = input.size();

    for (float value : input)
    {
        const std::uint16_t half = floatToHalf(value);

        encoded.bits_.writeBits(half, 16);
    }
    return encoded;
}

std::vector<float> FP16Quantizer::decode(const FP16EncodedTensor &encoded) const
{
    std::vector<float> output;

    output.reserve(encoded.value_count_);

    std::size_t bit_offset = 0;

    for (std::size_t i = 0; i < encoded.value_count_; ++i)
    {
        const std::uint16_t half = static_cast<std::uint16_t>(encoded.bits_.readBits(bit_offset, 16));
        bit_offset += 16;
        output.push_back(halfToFloat(half));
    }
    return output;
}
