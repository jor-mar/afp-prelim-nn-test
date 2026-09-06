#include "../include/int8.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

INT8Quantizer::INT8Quantizer(INT8Config config) : config_(config) {}

INT8EncodedTensor INT8Quantizer::encode(const std::vector<float> &input) const
{
    INT8EncodedTensor encoded;

    encoded.value_count_ = input.size();

    if (input.empty())
    {
        encoded.scale_ = 1.0f;

        return encoded;
    }

    float max_abs = 0.0f;

    for (float value : input)
    {
        max_abs = std::max(max_abs, std::abs(value));
    }

    if (max_abs == 0.0f)
    {
        encoded.scale_ = 1.0f;

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            encoded.bits_.writeBits(0, 8);
        }

        return encoded;
    }

    encoded.scale_ = max_abs / 127.0f;

    for (float value : input)
    {
        int quantized = static_cast<int>(std::round(value / encoded.scale_));

        // quantized = std::clamp(quantized, -127, 127);

        if (quantized < -127)
        {
            quantized = -127;
        }
        else if (quantized > 127)
        {
            quantized =127;
        }


        const std::int8_t int8_value = static_cast<std::int8_t>(quantized);

        const std::uint8_t raw_bits = static_cast<std::uint8_t>(int8_value);

        encoded.bits_.writeBits(raw_bits, 8);
    }

    return encoded;
}

std::vector<float> INT8Quantizer::decode(const INT8EncodedTensor &encoded) const
{
    std::vector<float> output;

    output.reserve(encoded.value_count_);

    std::size_t bit_offset = 0;

    for (std::size_t i = 0; i < encoded.value_count_; ++i)
    {
        const std::uint8_t raw_bits = static_cast<std::uint8_t>(encoded.bits_.readBits(bit_offset, 8));

        bit_offset += 8;

        const std::int8_t quantized = static_cast<std::int8_t>(raw_bits);

        const float value = static_cast<float>(quantized) * encoded.scale_;

        output.push_back(value);
    }

    return output;
}

const INT8Config &INT8Quantizer::getConfig() const
{
    return config_;
}