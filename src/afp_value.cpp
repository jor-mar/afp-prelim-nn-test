#include "../include/afp_value.hpp"
#include <stdexcept>
#include <algorithm>

namespace AFP {

// ============================================================================
// Utility Functions
// ============================================================================

namespace Utils {

int integerLog2(uint64_t value) {
    if (value == 0) return -1;
    int result = -1;
    while (value != 0) {
        value >>= 1;
        ++result;
    }
    return result;
}

uint64_t shiftRightRounded(uint64_t value, int shift) {
    if (shift <= 0) return value;
    if (shift >= 64) return 0;
    
    const uint64_t truncated = value >> shift;
    const uint64_t remainder = value & maskBits(shift);
    const uint64_t halfway = uint64_t{1} << (shift - 1);
    
    if (remainder > halfway) return truncated + 1;
    if (remainder == halfway && (truncated & 1U)) return truncated + 1;
    
    return truncated;
}

uint64_t maskBits(int bit_count) {
    if (bit_count <= 0) return 0;
    if (bit_count >= 64) return std::numeric_limits<uint64_t>::max();
    return (uint64_t{1} << bit_count) - 1;
}

int decodeSharedExponent(uint8_t encoded) {
    if (encoded == 0) return -126;
    return static_cast<int>(encoded) - 127;
}

uint8_t encodeSharedExponent(int exponent) {
    if (exponent < -126) exponent = -126;
    if (exponent > 127) exponent = 127;
    return static_cast<uint8_t>(exponent + 127);
}

void normalizeAccumulator(Accumulator& acc) {
    if (acc.zero || acc.significand == 0) {
        acc.zero = true;
        acc.negative = false;
        acc.significand = 0;
        acc.exponent = 0;
        return;
    }
    
    const uint64_t magnitude = static_cast<uint64_t>(acc.significand);
    constexpr int accumulator_bits = 48;
    const int highest_bit = integerLog2(magnitude);
    
    if (highest_bit >= accumulator_bits) {
        const int shift = highest_bit - accumulator_bits + 1;
        acc.significand = static_cast<int64_t>(shiftRightRounded(magnitude, shift));
        acc.exponent += shift;
    }
}

} // namespace Utils

// ============================================================================
// Value Implementation
// ============================================================================

bool Value::isZero() const {
    return offset == 7 && mantissa == 0;
}

bool Value::isNegative() const {
    return negative && !isZero();
}

bool Value::isPositive() const {
    return !negative && !isZero();
}

int Value::compare(const Value& other) const {
    const bool a_zero = isZero();
    const bool b_zero = other.isZero();
    
    if (a_zero && b_zero) return 0;
    if (a_zero) return other.negative ? 1 : -1;
    if (b_zero) return negative ? -1 : 1;
    
    if (negative != other.negative) return negative ? -1 : 1;
    
    const Product pa = toProduct();
    const Product pb = other.toProduct();
    
    const int a_top = pa.scale_exponent + Utils::integerLog2(pa.significand);
    const int b_top = pb.scale_exponent + Utils::integerLog2(pb.significand);
    
    if (a_top < b_top) return -1;
    if (a_top > b_top) return 1;
    
    if (pa.scale_exponent == pb.scale_exponent) {
        if (pa.significand < pb.significand) return -1;
        if (pa.significand > pb.significand) return 1;
        return 0;
    }
    
    const int shift = pa.scale_exponent - pb.scale_exponent;
    if (shift > 0) {
        if (shift >= 64) return 1;
        const uint64_t lhs = pa.significand << shift;
        return lhs < pb.significand ? -1 : (lhs > pb.significand ? 1 : 0);
    } else {
        const int reverse_shift = -shift;
        if (reverse_shift >= 64) return -1;
        const uint64_t rhs = pb.significand << reverse_shift;
        return pa.significand < rhs ? -1 : (pa.significand > rhs ? 1 : 0);
    }
}

Value Value::negate() const {
    if (isZero()) return *this;
    Value result = *this;
    result.negative = !result.negative;
    return result;
}

Value Value::absolute() const {
    Value result = *this;
    result.negative = false;
    return result;
}

Value Value::relu() const {
    if (isNegative()) return zero();
    return *this;
}

Value Value::scalePowerOfTwo(int power) const {
    if (isZero()) return *this;
    Value result = *this;
    int new_exponent = static_cast<int>(exponent) + power;
    if (new_exponent < -126) new_exponent = -126;
    if (new_exponent > 127) new_exponent = 127;
    result.exponent = static_cast<int8_t>(new_exponent);
    return result;
}

// ============================================================================
// Product Conversion
// ============================================================================

Product Value::toProduct() const {
    Product result;
    if (isZero()) return result;
    
    result.negative = negative;
    result.significand = mantissa;
    const int mant_bits = mantissaBits();
    result.scale_exponent = static_cast<int>(exponent) - static_cast<int>(offset) - mant_bits;
    result.zero = false;
    return result;
}

// ============================================================================
// AFP-Native Arithmetic Operations
// ============================================================================

Value Value::add(const Value& other) const {
    // Convert to products for normalization
    const Product prod_a = toProduct();
    const Product prod_b = other.toProduct();
    
    // Add products using accumulator
    Accumulator acc;
    
    // Add first product
    if (!prod_a.zero && prod_a.significand != 0) {
        acc.negative = prod_a.negative;
        acc.significand = static_cast<int64_t>(prod_a.significand);
        acc.exponent = prod_a.scale_exponent;
        acc.zero = false;
        Utils::normalizeAccumulator(acc);
    }
    
    // Add second product
    if (!prod_b.zero && prod_b.significand != 0) {
        if (acc.zero || acc.significand == 0) {
            acc.negative = prod_b.negative;
            acc.significand = static_cast<int64_t>(prod_b.significand);
            acc.exponent = prod_b.scale_exponent;
            acc.zero = false;
            Utils::normalizeAccumulator(acc);
        } else {
            const int common_exponent = std::min(acc.exponent, prod_b.scale_exponent);
            const int acc_shift = acc.exponent - common_exponent;
            const int prod_shift = prod_b.scale_exponent - common_exponent;
            
            int64_t acc_value = acc.significand;
            int64_t prod_value = static_cast<int64_t>(prod_b.significand);
            
            if (acc_shift > 0 && acc_shift < 63) acc_value <<= acc_shift;
            if (prod_shift > 0 && prod_shift < 63) prod_value <<= prod_shift;
            
            if (acc.negative) acc_value = -acc_value;
            if (prod_b.negative) prod_value = -prod_value;
            
            const int64_t sum = acc_value + prod_value;
            
            if (sum == 0) {
                return zero();
            }
            
            acc.negative = sum < 0;
            acc.significand = sum < 0 ? -sum : sum;
            acc.exponent = common_exponent;
            acc.zero = false;
            Utils::normalizeAccumulator(acc);
        }
    }
    
    if (acc.zero) return zero();
    
    const int new_exponent = acc.exponent + Utils::integerLog2(static_cast<uint64_t>(acc.significand));
    return fromAccumulator(acc, new_exponent, false);
}

Value Value::subtract(const Value& other) const {
    Value negated_other = other;
    if (!other.isZero()) negated_other.negative = !negated_other.negative;
    return add(negated_other);
}

Value Value::multiply(const Value& other) const {
    if (isZero() || other.isZero()) return zero();
    
    const Product prod_a = toProduct();
    const Product prod_b = other.toProduct();
    
    Product result;
    result.negative = prod_a.negative ^ prod_b.negative;
    result.significand = prod_a.significand * prod_b.significand;
    result.scale_exponent = prod_a.scale_exponent + prod_b.scale_exponent;
    result.zero = result.significand == 0;
    
    if (result.zero) return zero();
    
    Accumulator acc;
    acc.negative = result.negative;
    acc.significand = static_cast<int64_t>(result.significand);
    acc.exponent = result.scale_exponent;
    acc.zero = false;
    
    const int new_exponent = acc.exponent + Utils::integerLog2(static_cast<uint64_t>(acc.significand));
    return fromAccumulator(acc, new_exponent, false);
}

Value Value::divide(const Value& other) const {
    if (isZero()) return zero();
    if (other.isZero()) throw std::invalid_argument("AFP division by zero");
    
    const Product numerator = toProduct();
    const Product denominator = other.toProduct();
    
    if (numerator.zero) return zero();
    
    constexpr int division_precision_bits = 24;
    
    if (numerator.significand > (std::numeric_limits<uint64_t>::max() >> division_precision_bits)) {
        throw std::overflow_error("AFP division numerator overflow");
    }
    
    const uint64_t scaled_numerator = numerator.significand << division_precision_bits;
    const uint64_t quotient = scaled_numerator / denominator.significand;
    const uint64_t remainder = scaled_numerator % denominator.significand;
    
    uint64_t rounded_quotient = quotient;
    if (remainder >= denominator.significand - remainder) {
        if (rounded_quotient == std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error("AFP division quotient overflow");
        }
        ++rounded_quotient;
    }
    
    if (rounded_quotient == 0) return zero();
    
    Product result;
    result.negative = numerator.negative ^ denominator.negative;
    result.significand = rounded_quotient;
    result.scale_exponent = numerator.scale_exponent - denominator.scale_exponent - division_precision_bits;
    result.zero = false;
    
    Accumulator acc;
    acc.negative = result.negative;
    acc.significand = static_cast<int64_t>(result.significand);
    acc.exponent = result.scale_exponent;
    acc.zero = false;
    
    const int new_exponent = acc.exponent + Utils::integerLog2(static_cast<uint64_t>(acc.significand));
    return fromAccumulator(acc, new_exponent, false);
}

// ============================================================================
// Specialized Operations
// ============================================================================

Value Value::reciprocal() const {
    if (isZero()) throw std::invalid_argument("AFP reciprocal: division by zero");
    
    const Product p = toProduct();
    const int top = p.scale_exponent + Utils::integerLog2(p.significand);
    
    Value estimate;
    estimate.negative = negative;
    estimate.exponent = static_cast<int8_t>(-top);
    estimate.offset = 0;
    estimate.mantissa = static_cast<uint8_t>(32);
    
    const Value two = twoValue();
    
    // Newton-Raphson iteration
    for (int iteration = 0; iteration < 3; ++iteration) {
        const Value xy = multiply(estimate);
        const Value correction = two.subtract(xy);
        estimate = estimate.multiply(correction);
        if (estimate.isZero()) break;
    }
    
    return estimate;
}

Value Value::sqrt() const {
    if (isZero()) return zero();
    if (negative) throw std::invalid_argument("AFP sqrt: negative input");
    
    const Product p = toProduct();
    const int top = p.scale_exponent + Utils::integerLog2(p.significand);
    const int initial_exponent = top / 2;
    
    Value estimate;
    estimate.negative = false;
    estimate.exponent = static_cast<int8_t>(initial_exponent);
    estimate.offset = 0;
    estimate.mantissa = static_cast<uint8_t>(32);
    
    const Value half = halfValue();
    
    // Newton-Raphson iteration
    for (int iteration = 0; iteration < 4; ++iteration) {
        const Value quotient = divide(estimate);
        const Value sum = estimate.add(quotient);
        estimate = sum.multiply(half);
    }
    
    return estimate;
}

Value Value::reciprocalSqrt() const {
    const Value root = sqrt();
    return root.reciprocal();
}

Value Value::exp() const {
    if (isZero()) return oneValue();
    
    const Value one = oneValue();
    Value half;
    half.negative = false;
    half.exponent = -1;
    half.offset = 0;
    half.mantissa = static_cast<uint8_t>(32);
    
    Value one_sixth;
    one_sixth.negative = false;
    one_sixth.exponent = -3;
    one_sixth.offset = 0;
    one_sixth.mantissa = static_cast<uint8_t>(43);
    
    Value one_twenty_fourth;
    one_twenty_fourth.negative = false;
    one_twenty_fourth.exponent = -5;
    one_twenty_fourth.offset = 0;
    one_twenty_fourth.mantissa = static_cast<uint8_t>(43);
    
    Value one_one_twentieth;
    one_one_twentieth.negative = false;
    one_one_twentieth.exponent = -7;
    one_one_twentieth.offset = 0;
    one_one_twentieth.mantissa = static_cast<uint8_t>(43);
    
    Value one_seven_twentieth;
    one_seven_twentieth.negative = false;
    one_seven_twentieth.exponent = -9;
    one_seven_twentieth.offset = 0;
    one_seven_twentieth.mantissa = static_cast<uint8_t>(45);
    
    // Taylor series expansion: e^x = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + x^6/6!
    const Value x2 = multiply(*this);
    const Value x3 = x2.multiply(*this);
    const Value x4 = x3.multiply(*this);
    const Value x5 = x4.multiply(*this);
    const Value x6 = x5.multiply(*this);
    
    const Value term2 = x2.multiply(half);
    const Value term3 = x3.multiply(one_sixth);
    const Value term4 = x4.multiply(one_twenty_fourth);
    const Value term5 = x5.multiply(one_one_twentieth);
    const Value term6 = x6.multiply(one_seven_twentieth);
    
    Value result = one.add(*this);
    result = result.add(term2);
    result = result.add(term3);
    result = result.add(term4);
    result = result.add(term5);
    result = result.add(term6);
    
    return result;
}

Value Value::tanh() const {
    if (isZero()) return *this;
    
    const Product product = toProduct();
    const int highest = product.scale_exponent + Utils::integerLog2(product.significand);
    
    // Check if value is large enough to saturate
    bool large = false;
    if (highest > 1) {
        large = true;
    } else if (highest == 1) {
        const int shift = 1 - product.scale_exponent;
        if (shift >= 0 && shift < 64) {
            large = product.significand >= (uint64_t{3} << shift);
        }
    }
    
    if (large) {
        Value result;
        result.negative = negative;
        result.exponent = 0;
        result.offset = 0;
        result.mantissa = uint64_t{1} << 5;
        return result;
    }
    
    // tanh(x) = (27x + x^3) / (27 + 9x^2) using rational approximation
    const Value square = multiply(*this);
    
    // Numerator: 27x + x^3
    Value twenty_seven;
    twenty_seven.negative = false;
    twenty_seven.exponent = 4;
    twenty_seven.offset = 0;
    twenty_seven.mantissa = static_cast<uint8_t>(34);
    
    const Value numerator_part = twenty_seven.add(square);
    const Value numerator = multiply(numerator_part);
    
    // Denominator: 27 + 9x^2
    Product nine_square = square.toProduct();
    nine_square.significand *= 9;
    
    Accumulator denom_acc;
    denom_acc.negative = false;
    denom_acc.significand = 27;
    denom_acc.exponent = 0;
    denom_acc.zero = false;
    
    const int shift = nine_square.scale_exponent - denom_acc.exponent;
    if (shift >= 0 && shift < 64) {
        nine_square.significand <<= shift;
    } else if (shift < 0 && -shift < 64) {
        denom_acc.significand <<= -shift;
    }
    
    denom_acc.significand += static_cast<int64_t>(nine_square.significand);
    Utils::normalizeAccumulator(denom_acc);
    
    const Value denominator = fromAccumulator(denom_acc, 0, false);
    
    return numerator.divide(denominator);
}

// ============================================================================
// Conversion from Product/Accumulator
// ============================================================================

Value Value::fromProduct(const Product& prod, int shared_exponent, bool positive_field) {
    Accumulator acc;
    acc.negative = prod.negative;
    acc.significand = static_cast<int64_t>(prod.significand);
    acc.exponent = prod.scale_exponent;
    acc.zero = prod.zero;
    return fromAccumulator(acc, shared_exponent, positive_field);
}

Value Value::fromAccumulator(const Accumulator& acc, int shared_exponent, bool positive_field) {
    Value result;
    result.exponent = static_cast<int8_t>(shared_exponent);
    result.negative = acc.negative;
    
    const int mantissa_bits = positive_field ? 6 : 5;
    
    if (acc.zero || acc.significand == 0) {
        result.negative = false;
        result.offset = 7;
        result.mantissa = 0;
        return result;
    }
    
    const int highest_bit = Utils::integerLog2(static_cast<uint64_t>(acc.significand));
    const int value_exponent = acc.exponent + highest_bit;
    
    int offset = shared_exponent - value_exponent;
    if (offset < 0) offset = 0;
    if (offset > 7) offset = 7;
    
    result.offset = static_cast<uint8_t>(offset);
    
    const int target_exponent = shared_exponent - offset - mantissa_bits;
    const int shift = target_exponent - acc.exponent;
    
    uint64_t scaled;
    if (shift >= 0) {
        if (shift >= 64) scaled = 0;
        else scaled = Utils::shiftRightRounded(static_cast<uint64_t>(acc.significand), shift);
    } else {
        const int left_shift = -shift;
        if (left_shift >= 63) scaled = std::numeric_limits<uint64_t>::max();
        else scaled = static_cast<uint64_t>(acc.significand) << left_shift;
    }
    
    if (offset < 7) {
        const uint64_t implicit = uint64_t{1} << mantissa_bits;
        const uint64_t maximum = (implicit << 1) - 1;
        if (scaled < implicit) scaled = implicit;
        if (scaled > maximum) scaled = maximum;
        result.mantissa = static_cast<uint8_t>(scaled);
    } else {
        const uint64_t maximum = Utils::maskBits(mantissa_bits);
        if (scaled > maximum) scaled = maximum;
        result.mantissa = static_cast<uint8_t>(scaled);
    }
    
    if (result.isZero()) result.negative = false;
    
    return result;
}

} // namespace AFP