#include "../include/bf16.hpp"

#include <cstdint>
#include <cstring>

namespace
{
    std::uint32_t floatToBits(float value)
    {
        std::uint32_t bits;
        std::memcpy(&bits, &value,sizeof(float));
        return bits;
    }

    float bitsToFloat(std::uint32_t bits)
    {
        float value;
        std::memcpy(&value, &bits, sizeof(float));
        return value;
    }

}

BF16EncodedTensor BF16Quantizer::encode(const std::vector<float> &input) const
{
    BF16EncodedTensor encoded;

    encoded.value_count_ = input.size();

    for (float value : input)
    {
        const std::uint32_t fp32_bits = floatToBits(value);
        const std::uint32_t rounding_bias = 0x7FFFU + ((fp32_bits >> 16) & 1U);

        const std::uint32_t rounded = fp32_bits + rounding_bias;

        const std::uint16_t bf16_bits = static_cast<std::uint16_t>(rounded >> 16);

        encoded.bits_.writeBits(bf16_bits, 16);
    }
    return encoded;
}

std::vector<float> BF16Quantizer::decode(const BF16EncodedTensor &encoded) const
{
    std::vector<float> output;
    output.reserve(encoded.value_count_);
    std::size_t bit_offset = 0;
    for (std::size_t i = 0; i < encoded.value_count_; ++i)
    {
        const std::uint16_t bf16_bits = static_cast<std::uint16_t>(encoded.bits_.readBits(bit_offset, 16));

        bit_offset += 16;

        const std::uint32_t fp32_bits = static_cast<std::uint32_t>(bf16_bits) << 16;

        output.push_back(bitsToFloat(fp32_bits));
    }

    return output;
}