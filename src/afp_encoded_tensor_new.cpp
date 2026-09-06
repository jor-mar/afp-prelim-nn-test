#include "../include/afp_encoded_tensor_new.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
    constexpr uint32_t exponent_mask = 0x7F800000U;
    constexpr uint32_t fraction_mask = 0x007FFFFFU;
    
    uint32_t floatBits(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }
    
    uint64_t bitMask(int bit_count) {
        if (bit_count <= 0) return 0;
        return (uint64_t{1} << bit_count) - 1;
    }
    
    uint64_t roundShiftRight(uint64_t value, int shift) {
        if (shift <= 0) return value << -shift;
        if (shift >= 64) return 0;
        
        const uint64_t truncated = value >> shift;
        const uint64_t remainder = value & ((uint64_t{1} << shift) - 1);
        const uint64_t halfway = uint64_t{1} << (shift - 1);
        
        if (remainder > halfway) return truncated + 1;
        if (remainder == halfway && (truncated & 1)) return truncated + 1;
        
        return truncated;
    }
}

// ============================================================================
// AFPQuantizer Implementation
// ============================================================================

constexpr int AFPQuantizer::block_size;
constexpr int AFPQuantizer::half_block_size;
constexpr int AFPQuantizer::maximum_offset;
constexpr int AFPQuantizer::fp32_exponent_bias;
constexpr int AFPQuantizer::fp32_fraction_bits;

void AFPQuantizer::validateConfig(const AFPConfig& config) {
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

AFPQuantizer::AFPQuantizer(AFPConfig config) : config_(config) {
    validateConfig(config_);
}

int AFPQuantizer::unbiasedExponent(float value) {
    const uint32_t bits = floatBits(value);
    const int exponent = static_cast<int>((bits & exponent_mask) >> fp32_fraction_bits);
    return exponent == 0 ? -126 : exponent - fp32_exponent_bias;
}

uint32_t AFPQuantizer::significand(float value) {
    const uint32_t bits = floatBits(value);
    const uint32_t exponent = (bits & exponent_mask) >> fp32_fraction_bits;
    const uint32_t fraction = bits & fraction_mask;
    return exponent == 0 ? fraction : (uint32_t{1} << fp32_fraction_bits) | fraction;
}

bool AFPQuantizer::halfIsPositive(const std::vector<float>& input, std::size_t start, std::size_t end) {
    for (std::size_t i = start; i < end; ++i) {
        if (input[i] < 0.0f) return false;
    }
    return true;
}

int AFPQuantizer::sharedExponent(const std::vector<float>& block) {
    bool found_nonzero = false;
    int maximum = -126;
    
    for (float value : block) {
        if (value == 0.0f) continue;
        const int exponent = unbiasedExponent(std::fabs(value));
        if (!found_nonzero || exponent > maximum) {
            maximum = exponent;
            found_nonzero = true;
        }
    }
    
    return maximum;
}

uint8_t AFPQuantizer::encodeSharedExponent(int exponent) {
    if (exponent < -126 || exponent > 127)
        throw std::out_of_range("AFP shared exponent is outside FP32 normal range");
    return static_cast<uint8_t>(exponent + fp32_exponent_bias);
}

int AFPQuantizer::decodeSharedExponent(uint8_t exponent) {
    return exponent == 0 ? -126 : static_cast<int>(exponent) - fp32_exponent_bias;
}

uint64_t AFPQuantizer::encodeMantissa(float magnitude, int exponent, int offset, int mantissa_bits) {
    if (magnitude == 0.0f) return 0;
    
    const double scaled = std::ldexp(static_cast<double>(magnitude), -(exponent - offset));
    
    if (offset < maximum_offset) {
        const double fraction = scaled - 1.0;
        const double factor = static_cast<double>(uint64_t{1} << mantissa_bits);
        uint64_t mantissa = static_cast<uint64_t>(std::nearbyint(fraction * factor));
        const uint64_t maximum = bitMask(mantissa_bits);
        return mantissa > maximum ? maximum : mantissa;
    } else {
        const double factor = std::ldexp(1.0, exponent - maximum_offset - mantissa_bits);
        uint64_t mantissa = static_cast<uint64_t>(std::nearbyint(static_cast<double>(magnitude) / factor));
        const uint64_t maximum = bitMask(mantissa_bits);
        return mantissa > maximum ? maximum : mantissa;
    }
}

float AFPQuantizer::decodeValue(bool negative, int exponent, int offset, uint64_t mantissa, int mantissa_bits) {
    if (offset == maximum_offset && mantissa == 0) return 0.0f;
    
    double magnitude;
    if (offset < maximum_offset) {
        const double fraction = static_cast<double>(mantissa) / static_cast<double>(uint64_t{1} << mantissa_bits);
        magnitude = std::ldexp(1.0 + fraction, exponent - offset);
    } else {
        magnitude = std::ldexp(static_cast<double>(mantissa), exponent - maximum_offset - mantissa_bits);
    }
    
    const float result = static_cast<float>(magnitude);
    return negative ? -result : result;
}

AFPEncodedTensor AFPQuantizer::encode(const std::vector<float>& input) const {
    AFPEncodedTensor encoded;
    encoded.config_ = config_;
    encoded.value_count_ = input.size();
    
    if (input.empty()) return encoded;
    
    constexpr uint8_t first_half_positive_bit = 0;
    constexpr uint8_t second_half_positive_bit = 1;
    
    for (std::size_t block_start = 0; block_start < input.size(); block_start += block_size) {
        encoded.block_offsets_.push_back(encoded.bits_.bitSize());
        
        // Process block with minimal temporary storage
        const int exponent = [this, &input, block_start]() {
            int max_exp = -126;
            bool found = false;
            for (std::size_t i = 0; i < block_size && block_start + i < input.size(); ++i) {
                const float value = input[block_start + i];
                if (value == 0.0f) continue;
                const int exp = unbiasedExponent(std::fabs(value));
                if (!found || exp > max_exp) {
                    max_exp = exp;
                    found = true;
                }
            }
            return max_exp;
        }();
        
        const bool first_half_positive = [this, &input, block_start]() {
            for (std::size_t i = 0; i < half_block_size && block_start + i < input.size(); ++i) {
                if (input[block_start + i] < 0.0f) return false;
            }
            return true;
        }();
        
        const bool second_half_positive = [this, &input, block_start]() {
            for (std::size_t i = half_block_size; i < block_size && block_start + i < input.size(); ++i) {
                if (input[block_start + i] < 0.0f) return false;
            }
            return true;
        }();
        
        uint8_t characterization = 0;
        if (first_half_positive) characterization |= uint8_t{1} << first_half_positive_bit;
        if (second_half_positive) characterization |= uint8_t{1} << second_half_positive_bit;
        
        encoded.bits_.writeBits(encodeSharedExponent(exponent), 8);
        encoded.bits_.writeBits(characterization, 8);
        
        for (std::size_t i = 0; i < block_size; ++i) {
            const float value = (block_start + i < input.size()) ? input[block_start + i] : 0.0f;
            
            if (!std::isfinite(value))
                throw std::invalid_argument("AFP8 does not support NaN or infinity");
            
            const bool positive_half = i < half_block_size ? first_half_positive : second_half_positive;
            const int effective_mantissa_bits = config_.mantissa_bits + (positive_half ? 1 : 0);
            const bool negative = std::signbit(value) && value != 0.0f;
            
            int offset = maximum_offset;
            uint64_t mantissa = 0;
            
            if (value != 0.0f) {
                const int value_exponent = unbiasedExponent(std::fabs(value));
                const int true_offset = exponent - value_exponent;
                offset = std::clamp(true_offset, 0, maximum_offset);
                mantissa = encodeMantissa(std::fabs(value), exponent, offset, effective_mantissa_bits);
            }
            
            if (positive_half) {
                const uint64_t extra_bit = mantissa >> config_.mantissa_bits;
                const uint64_t stored_mantissa = mantissa & bitMask(config_.mantissa_bits);
                encoded.bits_.writeBits(extra_bit, 1);
                encoded.bits_.writeBits(offset, 3);
                encoded.bits_.writeBits(stored_mantissa, 5);
            } else {
                encoded.bits_.writeBits(negative ? 1 : 0, 1);
                encoded.bits_.writeBits(offset, 3);
                encoded.bits_.writeBits(mantissa, 5);
            }
        }
    }
    
    return encoded;
}

std::vector<float> AFPQuantizer::decode(const AFPEncodedTensor& encoded) const {
    validateConfig(encoded.config_);
    
    std::vector<float> output;
    output.reserve(encoded.value_count_);
    
    if (encoded.value_count_ == 0) return output;
    
    constexpr uint8_t first_half_positive_bit = 0;
    constexpr uint8_t second_half_positive_bit = 1;
    
    std::size_t bit_offset = 0;
    std::size_t values_decoded = 0;
    
    auto readBits = [&encoded, &bit_offset](int bit_count) -> uint64_t {
        const uint64_t value = encoded.bits_.readBits(bit_offset, static_cast<std::size_t>(bit_count));
        bit_offset += static_cast<std::size_t>(bit_count);
        return value;
    };
    
    while (values_decoded < encoded.value_count_) {
        const uint8_t stored_exponent = static_cast<uint8_t>(readBits(8));
        const int exponent = decodeSharedExponent(stored_exponent);
        
        const uint8_t characterization = static_cast<uint8_t>(readBits(8));
        const bool first_half_positive = (characterization & (uint8_t{1} << first_half_positive_bit)) != 0;
        const bool second_half_positive = (characterization & (uint8_t{1} << second_half_positive_bit)) != 0;
        
        for (std::size_t i = 0; i < block_size; ++i) {
            const bool positive_half = i < half_block_size ? first_half_positive : second_half_positive;
            
            const uint64_t first_field = readBits(1);
            const int offset = static_cast<int>(readBits(3));
            uint64_t mantissa = readBits(5);
            
            bool negative = false;
            int effective_mantissa_bits = config_.mantissa_bits;
            
            if (positive_half) {
                mantissa |= first_field << config_.mantissa_bits;
                effective_mantissa_bits++;
            } else {
                negative = first_field != 0;
            }
            
            if (values_decoded < encoded.value_count_) {
                output.push_back(decodeValue(negative, exponent, offset, mantissa, effective_mantissa_bits));
                ++values_decoded;
            }
        }
    }
    
    return output;
}

AFPEncodedTensor AFPQuantizer::load(
    const std::vector<std::uint8_t>& data,
    std::size_t bit_size,
    std::size_t value_count,
    const AFPConfig& config,
    const std::vector<std::size_t>& block_offsets) const
{
    AFPEncodedTensor result;
    result.config_ = config;
    result.value_count_ = value_count;
    result.block_offsets_ = block_offsets;
    result.bits_.loadData(data, bit_size);
    return result;
}