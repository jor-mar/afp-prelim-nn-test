#include "../include/afp_math_new.hpp"
#include <algorithm>
#include <stdexcept>

// Constants
constexpr std::size_t AFPArithmetic::block_size;
constexpr std::size_t AFPArithmetic::half_block_size;
constexpr int AFPArithmetic::maximum_offset;
constexpr int AFPArithmetic::fp32_exponent_bias;

// Helper function to create single-element vector
static std::vector<AFP::Value> makeSingleValue(AFP::Value value) {
    std::vector<AFP::Value> result;
    result.push_back(value);
    return result;
}

// ============================================================================
// AFP Value I/O Operations
// ============================================================================

AFP::Value AFPArithmetic::readAFPValue(
    const AFPEncodedTensor &tensor,
    std::size_t block_index,
    std::size_t value_index)
{
    constexpr std::size_t shared_exponent_bits = 8;
    constexpr std::size_t characterization_bits = 8;
    constexpr std::size_t first_field_bits = 1;
    constexpr std::size_t offset_bits = 3;
    constexpr std::size_t stored_mantissa_bits = 5;
    
    constexpr std::size_t block_header_bits = shared_exponent_bits + characterization_bits;
    constexpr std::size_t value_bits = first_field_bits + offset_bits + stored_mantissa_bits;
    
    constexpr uint8_t first_half_positive_bit = 0;
    constexpr uint8_t second_half_positive_bit = 1;
    
    const std::size_t block_base = tensor.block_offsets_[block_index];
    
    const uint8_t exponent_field = static_cast<uint8_t>(
        tensor.bits_.readBits(block_base, shared_exponent_bits));
    
    const int exponent = AFP::Utils::decodeSharedExponent(exponent_field);
    
    const uint8_t characterization = static_cast<uint8_t>(
        tensor.bits_.readBits(block_base + shared_exponent_bits, characterization_bits));
    
    const bool is_first_half = value_index < half_block_size;
    const bool positive_half = is_first_half ?
        (characterization & (uint8_t{1} << first_half_positive_bit)) != 0 :
        (characterization & (uint8_t{1} << second_half_positive_bit)) != 0;
    
    const std::size_t value_base = block_base + block_header_bits + value_index * value_bits;
    
    const uint8_t first_field = static_cast<uint8_t>(
        tensor.bits_.readBits(value_base, first_field_bits));
    
    const uint8_t offset = static_cast<uint8_t>(
        tensor.bits_.readBits(value_base + first_field_bits, offset_bits));
    
    uint8_t mantissa = static_cast<uint8_t>(
        tensor.bits_.readBits(value_base + first_field_bits + offset_bits, stored_mantissa_bits));
    
    AFP::Value result;
    result.exponent = static_cast<int8_t>(exponent);
    result.offset = offset;
    
    if (positive_half) {
        mantissa |= static_cast<uint8_t>(first_field << stored_mantissa_bits);
        result.negative = false;
        
        if (result.offset < maximum_offset) {
            result.mantissa = (uint64_t{1} << 6) + mantissa;
        } else {
            result.mantissa = mantissa;
        }
    } else {
        result.negative = first_field != 0;
        
        if (result.offset < maximum_offset) {
            result.mantissa = (uint64_t{1} << 5) + mantissa;
        } else {
            result.mantissa = mantissa;
        }
    }
    
    if (result.isZero()) result.negative = false;
    
    return result;
}

void AFPArithmetic::writeAFPValue(
    BitStream &bits,
    const AFP::Value &value,
    bool positive_field)
{
    constexpr std::size_t stored_mantissa_bits = 5;
    const int value_mantissa_bits = positive_field ? 6 : 5;
    
    uint64_t mantissa = 0;
    
    if (value.offset < maximum_offset) {
        const uint64_t implicit = uint64_t{1} << value_mantissa_bits;
        if (value.mantissa >= implicit) {
            mantissa = value.mantissa - implicit;
        }
    } else {
        mantissa = value.mantissa;
    }
    
    if (positive_field) {
        const uint64_t extra_bit = mantissa >> stored_mantissa_bits;
        const uint64_t stored_mantissa = mantissa & AFP::Utils::maskBits(stored_mantissa_bits);
        
        bits.writeBits(extra_bit, 1);
        bits.writeBits(static_cast<uint64_t>(value.offset), 3);
        bits.writeBits(stored_mantissa, stored_mantissa_bits);
    } else {
        bits.writeBits(value.negative ? 1 : 0, 1);
        bits.writeBits(static_cast<uint64_t>(value.offset), 3);
        bits.writeBits(mantissa & AFP::Utils::maskBits(stored_mantissa_bits), stored_mantissa_bits);
    }
}

// ============================================================================
// Block-Level Operations
// ============================================================================

int AFPArithmetic::computeSharedExponent(
    const std::vector<AFP::Value> &values,
    std::size_t block_start)
{
    int shared_exponent = -126;
    bool found_nonzero = false;
    
    for (std::size_t i = 0; i < block_size && block_start + i < values.size(); ++i) {
        const AFP::Value &value = values[block_start + i];
        if (value.isZero()) continue;
        
        const int effective_exponent = value.effectiveExponent();
        if (!found_nonzero || effective_exponent > shared_exponent) {
            shared_exponent = effective_exponent;
            found_nonzero = true;
        }
    }
    
    return shared_exponent;
}

bool AFPArithmetic::computeHalfPositive(
    const std::vector<AFP::Value> &values,
    std::size_t block_start,
    bool first_half)
{
    const std::size_t start = first_half ? 0 : half_block_size;
    const std::size_t end = first_half ? half_block_size : block_size;
    
    for (std::size_t i = start; i < end && block_start + i < values.size(); ++i) {
        const AFP::Value &value = values[block_start + i];
        if (!value.isZero() && value.negative) {
            return false;
        }
    }
    
    return true;
}

uint8_t AFPArithmetic::buildCharacterization(
    bool first_half_positive,
    bool second_half_positive)
{
    constexpr uint8_t first_half_positive_bit = 0;
    constexpr uint8_t second_half_positive_bit = 1;
    
    uint8_t characterization = 0;
    if (first_half_positive) {
        characterization |= (uint8_t{1} << first_half_positive_bit);
    }
    if (second_half_positive) {
        characterization |= (uint8_t{1} << second_half_positive_bit);
    }
    
    return characterization;
}

// ============================================================================
// Tensor Construction
// ============================================================================

AFPEncodedTensor AFPArithmetic::buildTensorFromAFPValues(
    const std::vector<AFP::Value> &values,
    const AFPConfig &config)
{
    AFPEncodedTensor result;
    result.config_ = config;
    result.value_count_ = values.size();
    
    if (values.empty()) return result;
    
    for (std::size_t block_start = 0; block_start < values.size(); block_start += block_size) {
        result.block_offsets_.push_back(result.bits_.bitSize());
        
        const int shared_exponent = computeSharedExponent(values, block_start);
        const bool first_half_positive = computeHalfPositive(values, block_start, true);
        const bool second_half_positive = computeHalfPositive(values, block_start, false);
        const uint8_t characterization = buildCharacterization(first_half_positive, second_half_positive);
        
        result.bits_.writeBits(AFP::Utils::encodeSharedExponent(shared_exponent), 8);
        result.bits_.writeBits(characterization, 8);
        
        for (std::size_t i = 0; i < block_size; ++i) {
            const bool positive_field = i < half_block_size ? first_half_positive : second_half_positive;
            
            AFP::Value value;
            if (block_start + i < values.size()) {
                const AFP::Product normalized = values[block_start + i].toProduct();
                AFP::Accumulator acc;
                acc.negative = normalized.negative;
                acc.significand = static_cast<int64_t>(normalized.significand);
                acc.exponent = normalized.scale_exponent;
                acc.zero = normalized.zero;
                value = AFP::Value::fromAccumulator(acc, shared_exponent, positive_field);
            } else {
                value.exponent = static_cast<int8_t>(shared_exponent);
                value.negative = false;
                value.offset = maximum_offset;
                value.mantissa = 0;
            }
            
            writeAFPValue(result.bits_, value, positive_field);
        }
    }
    
    return result;
}

// ============================================================================
// Utilities
// ============================================================================

void AFPArithmetic::validateCompatible(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument("AFP tensor size mismatch");
    }
    if (a.config_.block_size != b.config_.block_size) {
        throw std::invalid_argument("AFP block size mismatch");
    }
}

// ============================================================================
// Public Interface - Element-wise Operations
// ============================================================================

AFPEncodedTensor AFPArithmetic::add(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    return elementWiseOp(a, b, [](const AFP::Value& av, const AFP::Value& bv) {
        return av.add(bv);
    });
}

AFPEncodedTensor AFPArithmetic::subtract(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    return elementWiseOp(a, b, [](const AFP::Value& av, const AFP::Value& bv) {
        return av.subtract(bv);
    });
}

AFPEncodedTensor AFPArithmetic::multiply(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    return elementWiseOp(a, b, [](const AFP::Value& av, const AFP::Value& bv) {
        return av.multiply(bv);
    });
}

AFPEncodedTensor AFPArithmetic::divide(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    return elementWiseOp(a, b, [](const AFP::Value& av, const AFP::Value& bv) {
        return av.divide(bv);
    });
}

AFPEncodedTensor AFPArithmetic::negate(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.negate();
    });
}

AFPEncodedTensor AFPArithmetic::absolute(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.absolute();
    });
}

AFPEncodedTensor AFPArithmetic::minimum(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    return elementWiseOp(a, b, [](const AFP::Value& av, const AFP::Value& bv) {
        return av.compare(bv) <= 0 ? av : bv;
    });
}

AFPEncodedTensor AFPArithmetic::maximum(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    return elementWiseOp(a, b, [](const AFP::Value& av, const AFP::Value& bv) {
        return av.compare(bv) >= 0 ? av : bv;
    });
}

AFPEncodedTensor AFPArithmetic::clamp(
    const AFPEncodedTensor &input,
    const AFPEncodedTensor &min_value,
    const AFPEncodedTensor &max_value)
{
    if (min_value.size() != 1 || max_value.size() != 1) {
        throw std::invalid_argument("AFP clamp: min_value and max_value must contain one value");
    }
    
    const AFP::Value min_val = readAFPValue(min_value, 0, 0);
    const AFP::Value max_val = readAFPValue(max_value, 0, 0);
    
    if (min_val.compare(max_val) > 0) {
        throw std::invalid_argument("AFP clamp: min_value greater than max_value");
    }
    
    return elementWiseUnaryOp(input, [&](const AFP::Value& v) {
        if (v.compare(min_val) < 0) return min_val;
        if (v.compare(max_val) > 0) return max_val;
        return v;
    });
}

AFPEncodedTensor AFPArithmetic::leakyRelu(
    const AFPEncodedTensor &input,
    const AFPEncodedTensor &alpha)
{
    if (alpha.size() != 1) {
        throw std::invalid_argument("AFP leakyRelu: alpha must contain one value");
    }
    
    const AFP::Value alpha_value = readAFPValue(alpha, 0, 0);
    
    return elementWiseUnaryOp(input, [&](const AFP::Value& v) {
        if (v.isZero() || !v.negative) return v;
        return v.multiply(alpha_value);
    });
}

AFPEncodedTensor AFPArithmetic::silu(const AFPEncodedTensor &input)
{
    return multiply(input, sigmoid(input));
}

// ============================================================================
// Public Interface - Dot Product
// ============================================================================

AFPEncodedTensor AFPArithmetic::dotProduct(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    validateCompatible(a, b);
    
    if (a.size() == 0) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), a.config_);
    }
    
    AFP::Accumulator accumulator;
    
    for (std::size_t index = 0; index < a.size(); ++index) {
        const std::size_t block = index / block_size;
        const std::size_t position = index % block_size;
        
        const AFP::Value a_value = readAFPValue(a, block, position);
        const AFP::Value b_value = readAFPValue(b, block, position);
        
        const AFP::Product product = a_value.toProduct();
        const AFP::Product prod_b = b_value.toProduct();
        
        // Add products to accumulator
        AFP::Product combined;
        combined.negative = product.negative ^ prod_b.negative;
        combined.significand = product.significand * prod_b.significand;
        combined.scale_exponent = product.scale_exponent + prod_b.scale_exponent;
        combined.zero = combined.significand == 0;
        
        if (!combined.zero && combined.significand != 0) {
            if (accumulator.zero || accumulator.significand == 0) {
                accumulator.negative = combined.negative;
                accumulator.significand = static_cast<int64_t>(combined.significand);
                accumulator.exponent = combined.scale_exponent;
                accumulator.zero = false;
                AFP::Utils::normalizeAccumulator(accumulator);
            } else {
                const int common_exponent = std::min(accumulator.exponent, combined.scale_exponent);
                const int acc_shift = accumulator.exponent - common_exponent;
                const int prod_shift = combined.scale_exponent - common_exponent;
                
                int64_t acc_value = accumulator.significand;
                int64_t prod_value = static_cast<int64_t>(combined.significand);
                
                if (acc_shift > 0 && acc_shift < 63) acc_value <<= acc_shift;
                if (prod_shift > 0 && prod_shift < 63) prod_value <<= prod_shift;
                
                if (accumulator.negative) acc_value = -acc_value;
                if (combined.negative) prod_value = -prod_value;
                
                const int64_t sum = acc_value + prod_value;
                
                if (sum == 0) {
                    accumulator.zero = true;
                    accumulator.negative = false;
                    accumulator.significand = 0;
                    accumulator.exponent = 0;
                } else {
                    accumulator.negative = sum < 0;
                    accumulator.significand = sum < 0 ? -sum : sum;
                    accumulator.exponent = common_exponent;
                    accumulator.zero = false;
                    AFP::Utils::normalizeAccumulator(accumulator);
                }
            }
        }
    }
    
    if (accumulator.zero) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), a.config_);
    }
    
    const int exponent = accumulator.exponent + AFP::Utils::integerLog2(static_cast<uint64_t>(accumulator.significand));
    return buildTensorFromAFPValues(makeSingleValue(AFP::Value::fromAccumulator(accumulator, exponent, false)), a.config_);
}

// ============================================================================
// Public Interface - Matrix Operations
// ============================================================================

AFPEncodedTensor AFPArithmetic::matrixVectorMultiply(
    const AFPEncodedTensor &weights,
    const AFPEncodedTensor &input,
    std::size_t rows,
    std::size_t columns)
{
    if (weights.size() != rows * columns) {
        throw std::invalid_argument("Weight tensor size doesn't match dimensions");
    }
    if (input.size() != columns) {
        throw std::invalid_argument("Input tensor size doesn't match columns");
    }
    
    std::vector<AFP::Value> output;
    output.reserve(rows);
    
    for (std::size_t row = 0; row < rows; ++row) {
        AFP::Accumulator row_accumulator;
        
        for (std::size_t col = 0; col < columns; ++col) {
            const std::size_t weight_idx = row * columns + col;
            const AFP::Value weight_value = readAFPValue(weights, weight_idx / block_size, weight_idx % block_size);
            const AFP::Value input_value = readAFPValue(input, col / block_size, col % block_size);
            
            const AFP::Product product = weight_value.toProduct();
            const AFP::Product prod_input = input_value.toProduct();
            
            AFP::Product combined;
            combined.negative = product.negative ^ prod_input.negative;
            combined.significand = product.significand * prod_input.significand;
            combined.scale_exponent = product.scale_exponent + prod_input.scale_exponent;
            combined.zero = combined.significand == 0;
            
            if (!combined.zero && combined.significand != 0) {
                if (row_accumulator.zero || row_accumulator.significand == 0) {
                    row_accumulator.negative = combined.negative;
                    row_accumulator.significand = static_cast<int64_t>(combined.significand);
                    row_accumulator.exponent = combined.scale_exponent;
                    row_accumulator.zero = false;
                    AFP::Utils::normalizeAccumulator(row_accumulator);
                } else {
                    const int common_exponent = std::min(row_accumulator.exponent, combined.scale_exponent);
                    const int acc_shift = row_accumulator.exponent - common_exponent;
                    const int prod_shift = combined.scale_exponent - common_exponent;
                    
                    int64_t acc_value = row_accumulator.significand;
                    int64_t prod_value = static_cast<int64_t>(combined.significand);
                    
                    if (acc_shift > 0 && acc_shift < 63) acc_value <<= acc_shift;
                    if (prod_shift > 0 && prod_shift < 63) prod_value <<= prod_shift;
                    
                    if (row_accumulator.negative) acc_value = -acc_value;
                    if (combined.negative) prod_value = -prod_value;
                    
                    const int64_t sum = acc_value + prod_value;
                    
                    if (sum == 0) {
                        row_accumulator.zero = true;
                        row_accumulator.negative = false;
                        row_accumulator.significand = 0;
                        row_accumulator.exponent = 0;
                    } else {
                        row_accumulator.negative = sum < 0;
                        row_accumulator.significand = sum < 0 ? -sum : sum;
                        row_accumulator.exponent = common_exponent;
                        row_accumulator.zero = false;
                        AFP::Utils::normalizeAccumulator(row_accumulator);
                    }
                }
            }
        }
        
        if (row_accumulator.zero) {
            output.push_back(AFP::Value::zero());
        } else {
            const int exponent = row_accumulator.exponent + AFP::Utils::integerLog2(static_cast<uint64_t>(row_accumulator.significand));
            output.push_back(AFP::Value::fromAccumulator(row_accumulator, exponent, false));
        }
    }
    
    return buildTensorFromAFPValues(output, weights.config_);
}

AFPEncodedTensor AFPArithmetic::matrixMultiply(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b,
    std::size_t rows_a,
    std::size_t cols_a,
    std::size_t cols_b)
{
    if (a.size() != rows_a * cols_a) {
        throw std::invalid_argument("Matrix A size doesn't match dimensions");
    }
    if (b.size() != cols_a * cols_b) {
        throw std::invalid_argument("Matrix B size doesn't match dimensions");
    }
    
    std::vector<AFP::Value> output;
    output.reserve(rows_a * cols_b);
    
    for (std::size_t i = 0; i < rows_a; ++i) {
        for (std::size_t j = 0; j < cols_b; ++j) {
            AFP::Accumulator element_accumulator;
            
            for (std::size_t k = 0; k < cols_a; ++k) {
                const std::size_t a_idx = i * cols_a + k;
                const std::size_t b_idx = k * cols_b + j;
                
                const AFP::Value a_value = readAFPValue(a, a_idx / block_size, a_idx % block_size);
                const AFP::Value b_value = readAFPValue(b, b_idx / block_size, b_idx % block_size);
                
                const AFP::Product product = a_value.toProduct();
                const AFP::Product prod_b = b_value.toProduct();
                
                AFP::Product combined;
                combined.negative = product.negative ^ prod_b.negative;
                combined.significand = product.significand * prod_b.significand;
                combined.scale_exponent = product.scale_exponent + prod_b.scale_exponent;
                combined.zero = combined.significand == 0;
                
                if (!combined.zero && combined.significand != 0) {
                    if (element_accumulator.zero || element_accumulator.significand == 0) {
                        element_accumulator.negative = combined.negative;
                        element_accumulator.significand = static_cast<int64_t>(combined.significand);
                        element_accumulator.exponent = combined.scale_exponent;
                        element_accumulator.zero = false;
                        AFP::Utils::normalizeAccumulator(element_accumulator);
                    } else {
                        const int common_exponent = std::min(element_accumulator.exponent, combined.scale_exponent);
                        const int acc_shift = element_accumulator.exponent - common_exponent;
                        const int prod_shift = combined.scale_exponent - common_exponent;
                        
                        int64_t acc_value = element_accumulator.significand;
                        int64_t prod_value = static_cast<int64_t>(combined.significand);
                        
                        if (acc_shift > 0 && acc_shift < 63) acc_value <<= acc_shift;
                        if (prod_shift > 0 && prod_shift < 63) prod_value <<= prod_shift;
                        
                        if (element_accumulator.negative) acc_value = -acc_value;
                        if (combined.negative) prod_value = -prod_value;
                        
                        const int64_t sum = acc_value + prod_value;
                        
                        if (sum == 0) {
                            element_accumulator.zero = true;
                            element_accumulator.negative = false;
                            element_accumulator.significand = 0;
                            element_accumulator.exponent = 0;
                        } else {
                            element_accumulator.negative = sum < 0;
                            element_accumulator.significand = sum < 0 ? -sum : sum;
                            element_accumulator.exponent = common_exponent;
                            element_accumulator.zero = false;
                            AFP::Utils::normalizeAccumulator(element_accumulator);
                        }
                    }
                }
            }
            
            if (element_accumulator.zero) {
                output.push_back(AFP::Value::zero());
            } else {
                const int exponent = element_accumulator.exponent + AFP::Utils::integerLog2(static_cast<uint64_t>(element_accumulator.significand));
                output.push_back(AFP::Value::fromAccumulator(element_accumulator, exponent, false));
            }
        }
    }
    
    return buildTensorFromAFPValues(output, a.config_);
}

// ============================================================================
// Public Interface - Activation Functions
// ============================================================================

AFPEncodedTensor AFPArithmetic::relu(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.relu();
    });
}

AFPEncodedTensor AFPArithmetic::sigmoid(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        if (v.isZero()) {
            AFP::Value result;
            result.negative = false;
            result.exponent = -1;
            result.offset = 0;
            result.mantissa = 32;
            return result;
        }
        
        const AFP::Product prod = v.toProduct();
        if (prod.scale_exponent >= 2) {
            AFP::Value result;
            result.negative = v.negative;
            result.exponent = v.negative ? -1 : 0;
            result.offset = v.negative ? 7 : 0;
            result.mantissa = v.negative ? 0 : 32;
            return result;
        }
        
        // Simple sigmoid approximation: 0.5 + 0.25*x - 1/48*x^3
        AFP::Value half;
        half.negative = false;
        half.exponent = -1;
        half.offset = 0;
        half.mantissa = 32;
        
        AFP::Value quarter;
        quarter.negative = false;
        quarter.exponent = -2;
        quarter.offset = 0;
        quarter.mantissa = 32;
        
        AFP::Value inverse48;
        inverse48.negative = false;
        inverse48.exponent = -6;
        inverse48.offset = 0;
        inverse48.mantissa = 43;
        
        const AFP::Value x2 = v.multiply(v);
        AFP::Value x3 = x2.multiply(v);
        x3 = x3.multiply(inverse48);
        
        AFP::Value result = half.add(v.multiply(quarter));
        x3.negative = !x3.negative;
        result = result.add(x3);
        
        return result;
    });
}

AFPEncodedTensor AFPArithmetic::tanh(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.tanh();
    });
}

AFPEncodedTensor AFPArithmetic::gelu(const AFPEncodedTensor &input)
{
    const AFP::Value half = AFP::Value::half();
    
    AFP::Value three_quarters;
    three_quarters.negative = false;
    three_quarters.exponent = -1;
    three_quarters.offset = 0;
    three_quarters.mantissa = (uint64_t{1} << 5) + 16;
    
    AFP::Value one_sixteenth;
    one_sixteenth.negative = false;
    one_sixteenth.exponent = -4;
    one_sixteenth.offset = 0;
    one_sixteenth.mantissa = uint64_t{1} << 5;
    
    const AFP::Value one = AFP::Value::one();
    
    return elementWiseUnaryOp(input, [&](const AFP::Value& x) {
        const AFP::Value x2 = x.multiply(x);
        const AFP::Value x3 = x2.multiply(x);
        const AFP::Value cubic = x3.multiply(one_sixteenth);
        const AFP::Value inner = x.add(cubic);
        const AFP::Value scaled = inner.multiply(three_quarters);
        const AFP::Value t = scaled.tanh();
        const AFP::Value one_plus_t = one.add(t);
        const AFP::Value half_x = x.multiply(half);
        return half_x.multiply(one_plus_t);
    });
}

AFPEncodedTensor AFPArithmetic::exp(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.exp();
    });
}

AFPEncodedTensor AFPArithmetic::softmax(const AFPEncodedTensor &input)
{
    if (input.size() == 0) return input;
    
    AFP::Value maximum_value = readAFPValue(input, 0, 0);
    for (std::size_t i = 1; i < input.size(); ++i) {
        const AFP::Value value = readAFPValue(input, i / block_size, i % block_size);
        if (value.compare(maximum_value) > 0) {
            maximum_value = value;
        }
    }
    
    std::vector<AFP::Value> exponentials;
    exponentials.reserve(input.size());
    
    AFP::Accumulator denominator;
    
    for (std::size_t i = 0; i < input.size(); ++i) {
        const AFP::Value value = readAFPValue(input, i / block_size, i % block_size);
        const AFP::Value shifted = value.subtract(maximum_value);
        const AFP::Value e = shifted.exp();
        exponentials.push_back(e);
        
        const AFP::Product product = e.toProduct();
        if (!product.zero && product.significand != 0) {
            if (denominator.zero || denominator.significand == 0) {
                denominator.negative = product.negative;
                denominator.significand = static_cast<int64_t>(product.significand);
                denominator.exponent = product.scale_exponent;
                denominator.zero = false;
                AFP::Utils::normalizeAccumulator(denominator);
            } else {
                const int common_exponent = std::min(denominator.exponent, product.scale_exponent);
                const int acc_shift = denominator.exponent - common_exponent;
                const int prod_shift = product.scale_exponent - common_exponent;
                
                int64_t acc_value = denominator.significand;
                int64_t prod_value = static_cast<int64_t>(product.significand);
                
                if (acc_shift > 0 && acc_shift < 63) acc_value <<= acc_shift;
                if (prod_shift > 0 && prod_shift < 63) prod_value <<= prod_shift;
                
                if (denominator.negative) acc_value = -acc_value;
                if (product.negative) prod_value = -prod_value;
                
                const int64_t sum = acc_value + prod_value;
                
                if (sum == 0) {
                    denominator.zero = true;
                    denominator.negative = false;
                    denominator.significand = 0;
                    denominator.exponent = 0;
                } else {
                    denominator.negative = sum < 0;
                    denominator.significand = sum < 0 ? -sum : sum;
                    denominator.exponent = common_exponent;
                    denominator.zero = false;
                    AFP::Utils::normalizeAccumulator(denominator);
                }
            }
        }
    }
    
    if (denominator.zero) {
        return buildTensorFromAFPValues(std::vector<AFP::Value>(input.size(), AFP::Value::zero()), input.config_);
    }
    
    const int denominator_exponent = denominator.exponent + AFP::Utils::integerLog2(static_cast<uint64_t>(denominator.significand));
    const AFP::Value denominator_value = AFP::Value::fromAccumulator(denominator, denominator_exponent, false);
    
    std::vector<AFP::Value> output;
    output.reserve(input.size());
    
    for (const AFP::Value& e : exponentials) {
        output.push_back(e.divide(denominator_value));
    }
    
    return buildTensorFromAFPValues(output, input.config_);
}

// ============================================================================
// Public Interface - Reduction Operations
// ============================================================================

AFPEncodedTensor AFPArithmetic::sum(const AFPEncodedTensor &input)
{
    if (input.size() == 0) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), input.config_);
    }
    
    AFP::Accumulator accumulator;
    
    for (std::size_t index = 0; index < input.size(); ++index) {
        const std::size_t block = index / block_size;
        const std::size_t position = index % block_size;
        
        const AFP::Value value = readAFPValue(input, block, position);
        const AFP::Product product = value.toProduct();
        
        if (!product.zero && product.significand != 0) {
            if (accumulator.zero || accumulator.significand == 0) {
                accumulator.negative = product.negative;
                accumulator.significand = static_cast<int64_t>(product.significand);
                accumulator.exponent = product.scale_exponent;
                accumulator.zero = false;
                AFP::Utils::normalizeAccumulator(accumulator);
            } else {
                const int common_exponent = std::min(accumulator.exponent, product.scale_exponent);
                const int acc_shift = accumulator.exponent - common_exponent;
                const int prod_shift = product.scale_exponent - common_exponent;
                
                int64_t acc_value = accumulator.significand;
                int64_t prod_value = static_cast<int64_t>(product.significand);
                
                if (acc_shift > 0 && acc_shift < 63) acc_value <<= acc_shift;
                if (prod_shift > 0 && prod_shift < 63) prod_value <<= prod_shift;
                
                if (accumulator.negative) acc_value = -acc_value;
                if (product.negative) prod_value = -prod_value;
                
                const int64_t sum = acc_value + prod_value;
                
                if (sum == 0) {
                    accumulator.zero = true;
                    accumulator.negative = false;
                    accumulator.significand = 0;
                    accumulator.exponent = 0;
                } else {
                    accumulator.negative = sum < 0;
                    accumulator.significand = sum < 0 ? -sum : sum;
                    accumulator.exponent = common_exponent;
                    accumulator.zero = false;
                    AFP::Utils::normalizeAccumulator(accumulator);
                }
            }
        }
    }
    
    if (accumulator.zero) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), input.config_);
    }
    
    const int exponent = accumulator.exponent + AFP::Utils::integerLog2(static_cast<uint64_t>(accumulator.significand));
    return buildTensorFromAFPValues(std::vector<AFP::Value>{AFP::Value::fromAccumulator(accumulator, exponent, false)}, input.config_);
}

AFPEncodedTensor AFPArithmetic::mean(const AFPEncodedTensor &input)
{
    if (input.size() == 0) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), input.config_);
    }
    
    const AFPEncodedTensor sum_tensor = sum(input);
    const AFP::Value sum_value = readAFPValue(sum_tensor, 0, 0);
    
    AFP::Value count;
    count.negative = false;
    count.offset = 0;
    
    const std::size_t n = input.size();
    const int n_exponent = AFP::Utils::integerLog2(static_cast<uint64_t>(n));
    count.exponent = static_cast<int8_t>(n_exponent);
    
    const uint64_t normalized_n = static_cast<uint64_t>(n) << 5;
    count.mantissa = static_cast<uint8_t>(AFP::Utils::shiftRightRounded(normalized_n, n_exponent));
    if (count.mantissa == 0) count.mantissa = 1;
    
    const AFP::Value mean_value = sum_value.divide(count);
    return buildTensorFromAFPValues(makeSingleValue(mean_value), input.config_);
}

AFPEncodedTensor AFPArithmetic::max(const AFPEncodedTensor &input)
{
    if (input.size() == 0) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), input.config_);
    }
    
    AFP::Value max_value = readAFPValue(input, 0, 0);
    for (std::size_t i = 1; i < input.size(); ++i) {
        const AFP::Value current = readAFPValue(input, i / block_size, i % block_size);
        if (current.compare(max_value) > 0) {
            max_value = current;
        }
    }
    
    return buildTensorFromAFPValues(makeSingleValue(max_value), input.config_);
}

AFPEncodedTensor AFPArithmetic::sumSquares(const AFPEncodedTensor &input)
{
    AFP::Accumulator accumulator;
    
    for (std::size_t i = 0; i < input.size(); ++i) {
        const AFP::Value value = readAFPValue(input, i / block_size, i % block_size);
        const AFP::Product square = value.toProduct();
        
        AFP::Product squared;
        squared.negative = false;
        squared.significand = square.significand * square.significand;
        squared.scale_exponent = square.scale_exponent * 2;
        squared.zero = squared.significand == 0;
        
        if (!squared.zero && squared.significand != 0) {
            if (accumulator.zero || accumulator.significand == 0) {
                accumulator.negative = squared.negative;
                accumulator.significand = static_cast<int64_t>(squared.significand);
                accumulator.exponent = squared.scale_exponent;
                accumulator.zero = false;
                AFP::Utils::normalizeAccumulator(accumulator);
            } else {
                const int common_exponent = std::min(accumulator.exponent, squared.scale_exponent);
                const int acc_shift = accumulator.exponent - common_exponent;
                const int prod_shift = squared.scale_exponent - common_exponent;
                
                int64_t acc_value = accumulator.significand;
                int64_t prod_value = static_cast<int64_t>(squared.significand);
                
                if (acc_shift > 0 && acc_shift < 63) acc_value <<= acc_shift;
                if (prod_shift > 0 && prod_shift < 63) prod_value <<= prod_shift;
                
                if (accumulator.negative) acc_value = -acc_value;
                if (squared.negative) prod_value = -prod_value;
                
                const int64_t sum = acc_value + prod_value;
                
                if (sum == 0) {
                    accumulator.zero = true;
                    accumulator.negative = false;
                    accumulator.significand = 0;
                    accumulator.exponent = 0;
                } else {
                    accumulator.negative = sum < 0;
                    accumulator.significand = sum < 0 ? -sum : sum;
                    accumulator.exponent = common_exponent;
                    accumulator.zero = false;
                    AFP::Utils::normalizeAccumulator(accumulator);
                }
            }
        }
    }
    
    if (accumulator.zero) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), input.config_);
    }
    
    const int exponent = accumulator.exponent + AFP::Utils::integerLog2(static_cast<uint64_t>(accumulator.significand));
    return buildTensorFromAFPValues(std::vector<AFP::Value>{AFP::Value::fromAccumulator(accumulator, exponent, false)}, input.config_);
}

AFPEncodedTensor AFPArithmetic::rms(const AFPEncodedTensor &input)
{
    if (input.size() == 0) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), input.config_);
    }
    
    const AFPEncodedTensor sum_squares_tensor = sumSquares(input);
    const AFP::Value total = readAFPValue(sum_squares_tensor, 0, 0);
    
    std::size_t n = input.size();
    int log_n = 0;
    while ((std::size_t{1} << log_n) < n) ++log_n;
    
    const AFP::Value scaled = total.scalePowerOfTwo(-log_n);
    const AFP::Value result = scaled.sqrt();
    
    return buildTensorFromAFPValues(makeSingleValue(result), input.config_);
}

// ============================================================================
// Public Interface - Element-wise Mathematical Operations
// ============================================================================

AFPEncodedTensor AFPArithmetic::reciprocal(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.reciprocal();
    });
}

AFPEncodedTensor AFPArithmetic::sqrt(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.sqrt();
    });
}

AFPEncodedTensor AFPArithmetic::reciprocalSqrt(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.reciprocalSqrt();
    });
}

// ============================================================================
// Public Interface - Normalization
// ============================================================================

AFPEncodedTensor AFPArithmetic::rmsNorm(const AFPEncodedTensor &input)
{
    if (input.size() == 0) return input;
    
    const AFPEncodedTensor rms_tensor = rms(input);
    const AFP::Value rms_value = readAFPValue(rms_tensor, 0, 0);
    
    if (rms_value.isZero()) return input;
    
    return elementWiseUnaryOp(input, [&](const AFP::Value& v) {
        return v.divide(rms_value);
    });
}

AFPEncodedTensor AFPArithmetic::layerNorm(const AFPEncodedTensor &input)
{
    if (input.size() == 0) return input;
    
    const AFPEncodedTensor mean_tensor = mean(input);
    const AFP::Value mean_value = readAFPValue(mean_tensor, 0, 0);
    
    std::size_t n = input.size();
    int log_n = 0;
    while ((std::size_t{1} << log_n) < n) ++log_n;
    
    const AFP::Value scaled_mean = mean_value.scalePowerOfTwo(-log_n);
    
    AFP::Accumulator variance_accumulator;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const AFP::Value value = readAFPValue(input, i / block_size, i % block_size);
        const AFP::Value centered = value.subtract(scaled_mean);
        const AFP::Product square = centered.toProduct();
        
        AFP::Product squared;
        squared.negative = false;
        squared.significand = square.significand * square.significand;
        squared.scale_exponent = square.scale_exponent * 2;
        squared.zero = squared.significand == 0;
        
        if (!squared.zero && squared.significand != 0) {
            if (variance_accumulator.zero || variance_accumulator.significand == 0) {
                variance_accumulator.negative = squared.negative;
                variance_accumulator.significand = static_cast<int64_t>(squared.significand);
                variance_accumulator.exponent = squared.scale_exponent;
                variance_accumulator.zero = false;
                AFP::Utils::normalizeAccumulator(variance_accumulator);
            } else {
                const int common_exponent = std::min(variance_accumulator.exponent, squared.scale_exponent);
                const int acc_shift = variance_accumulator.exponent - common_exponent;
                const int prod_shift = squared.scale_exponent - common_exponent;
                
                int64_t acc_value = variance_accumulator.significand;
                int64_t prod_value = static_cast<int64_t>(squared.significand);
                
                if (acc_shift > 0 && acc_shift < 63) acc_value <<= acc_shift;
                if (prod_shift > 0 && prod_shift < 63) prod_value <<= prod_shift;
                
                if (variance_accumulator.negative) acc_value = -acc_value;
                if (squared.negative) prod_value = -prod_value;
                
                const int64_t sum = acc_value + prod_value;
                
                if (sum == 0) {
                    variance_accumulator.zero = true;
                    variance_accumulator.negative = false;
                    variance_accumulator.significand = 0;
                    variance_accumulator.exponent = 0;
                } else {
                    variance_accumulator.negative = sum < 0;
                    variance_accumulator.significand = sum < 0 ? -sum : sum;
                    variance_accumulator.exponent = common_exponent;
                    variance_accumulator.zero = false;
                    AFP::Utils::normalizeAccumulator(variance_accumulator);
                }
            }
        }
    }
    
    if (variance_accumulator.zero) {
        return buildTensorFromAFPValues(std::vector<AFP::Value>(input.size(), AFP::Value::zero()), input.config_);
    }
    
    const int variance_exponent = variance_accumulator.exponent + AFP::Utils::integerLog2(static_cast<uint64_t>(variance_accumulator.significand));
    AFP::Value variance = AFP::Value::fromAccumulator(variance_accumulator, variance_exponent, false);
    variance = variance.scalePowerOfTwo(-log_n);
    
    AFP::Value epsilon;
    epsilon.negative = false;
    epsilon.exponent = -10;
    epsilon.offset = 0;
    epsilon.mantissa = uint64_t{1} << 5;
    
    const AFP::Value variance_with_epsilon = variance.add(epsilon);
    const AFP::Value denominator = variance_with_epsilon.sqrt();
    
    if (denominator.isZero()) {
        return buildTensorFromAFPValues(std::vector<AFP::Value>(input.size(), AFP::Value::zero()), input.config_);
    }
    
    return elementWiseUnaryOp(input, [&](const AFP::Value& v) {
        const AFP::Value centered = v.subtract(scaled_mean);
        return centered.divide(denominator);
    });
}

// ============================================================================
// Public Interface - Tensor Shape Operations
// ============================================================================

AFPEncodedTensor AFPArithmetic::transpose(
    const AFPEncodedTensor &input,
    std::size_t rows,
    std::size_t columns)
{
    if (input.size() != rows * columns) {
        throw std::invalid_argument("AFP transpose: tensor size does not match dimensions");
    }
    
    std::vector<AFP::Value> output;
    output.reserve(input.size());
    
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t source_index = row * columns + column;
            output.push_back(readAFPValue(input, source_index / block_size, source_index % block_size));
        }
    }
    
    return buildTensorFromAFPValues(output, input.config_);
}

AFPEncodedTensor AFPArithmetic::outerProduct(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    std::vector<AFP::Value> output;
    output.reserve(a.size() * b.size());
    
    for (std::size_t i = 0; i < a.size(); ++i) {
        const AFP::Value av = readAFPValue(a, i / block_size, i % block_size);
        for (std::size_t j = 0; j < b.size(); ++j) {
            const AFP::Value bv = readAFPValue(b, j / block_size, j % block_size);
            output.push_back(av.multiply(bv));
        }
    }
    
    return buildTensorFromAFPValues(output, a.config_);
}

// ============================================================================
// Public Interface - Broadcasting Operations
// ============================================================================

AFPEncodedTensor AFPArithmetic::broadcastAdd(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b,
    std::size_t target_size)
{
    if ((a.size() != target_size && a.size() != 1) || (b.size() != target_size && b.size() != 1)) {
        throw std::invalid_argument("AFP broadcast add: invalid tensor sizes");
    }
    if (a.config_.block_size != b.config_.block_size) {
        throw std::invalid_argument("AFP block size mismatch");
    }
    
    std::vector<AFP::Value> output;
    output.reserve(target_size);
    
    for (std::size_t i = 0; i < target_size; ++i) {
        const std::size_t a_idx = a.size() == 1 ? 0 : i;
        const std::size_t b_idx = b.size() == 1 ? 0 : i;
        
        const AFP::Value a_val = readAFPValue(a, a_idx / block_size, a_idx % block_size);
        const AFP::Value b_val = readAFPValue(b, b_idx / block_size, b_idx % block_size);
        output.push_back(a_val.add(b_val));
    }
    
    return buildTensorFromAFPValues(output, a.config_);
}

AFPEncodedTensor AFPArithmetic::broadcastMultiply(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b,
    std::size_t target_size)
{
    if ((a.size() != target_size && a.size() != 1) || (b.size() != target_size && b.size() != 1)) {
        throw std::invalid_argument("AFP broadcast multiply: invalid tensor sizes");
    }
    if (a.config_.block_size != b.config_.block_size) {
        throw std::invalid_argument("AFP block size mismatch");
    }
    
    std::vector<AFP::Value> output;
    output.reserve(target_size);
    
    for (std::size_t i = 0; i < target_size; ++i) {
        const std::size_t a_idx = a.size() == 1 ? 0 : i;
        const std::size_t b_idx = b.size() == 1 ? 0 : i;
        
        const AFP::Value a_val = readAFPValue(a, a_idx / block_size, a_idx % block_size);
        const AFP::Value b_val = readAFPValue(b, b_idx / block_size, b_idx % block_size);
        output.push_back(a_val.multiply(b_val));
    }
    
    return buildTensorFromAFPValues(output, a.config_);
}