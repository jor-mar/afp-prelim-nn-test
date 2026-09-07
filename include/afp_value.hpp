#pragma once

#include <cstdint>
#include <limits>

namespace AFP {

// Forward declarations
class Product;
class Accumulator;

/**
 * AFPValue represents a single value in AFP8 format.
 * All arithmetic operations are performed using AFP-native computations
 * without converting to floating-point representation.
 *
 * AFP-NATIVE ARITHMETIC CONTRACT
 * ------------------------------
 * A Value carries (negative, exponent, offset, mantissa) with an integer
 * mantissa. Arithmetic never touches float or double:
 *
 *   - toProduct()      re-expresses the value as an integer significand
 *                      plus an integer scale exponent (value = significand
 *                      * 2^scale_exponent, sign carried separately).
 *   - Accumulator      accumulates int64 significands at a common integer
 *                      scale exponent (Utils::normalizeAccumulator).
 *   - add/subtract/    integer shifts, multiplies and divides only.
 *     multiply/divide
 *   - sqrt/exp/tanh/   AFP-native iterations on Value/Product/Accumulator
 *     sigmoid/...      with hard-coded AFP constants (e.g. Value::half()).
 *
 * The ONLY sanctioned float conversions in the library live in the codec
 * (AFPQuantizer::encode / AFPQuantizer::decode), which translate between
 * the outside world's FP32 tensors and AFP. Any operation that converts
 * an AFP value to float/double in order to compute and converts the
 * result back violates this contract.
 */
class Value {
public:
    // Constructors
    constexpr Value() : negative(false), exponent(-126), offset(7), mantissa(0), positive_field(false) {}
    
    constexpr Value(bool neg, int8_t exp, uint8_t off, uint8_t mant, bool positive = false)
        : negative(neg), exponent(exp), offset(off), mantissa(mant), positive_field(positive) {}
    
    // Static factory methods for common constants
    static constexpr Value zero() {
        return Value(false, -126, 7, 0);
    }
    
    static constexpr Value one() {
        return Value(false, 0, 0, 32);
    }
    
    static constexpr Value two() {
        return Value(false, 1, 0, 32);
    }
    
    static constexpr Value half() {
        return Value(false, -1, 0, 32);
    }
    
    // Helper methods for internal use (non-constexpr for flexibility)
    static Value oneValue() { return one(); }
    static Value twoValue() { return two(); }
    static Value halfValue() { return half(); }
    
    // AFP-native arithmetic operations
    Value add(const Value& other) const;
    Value subtract(const Value& other) const;
    Value multiply(const Value& other) const;
    Value divide(const Value& other) const;
    Value negate() const;
    Value absolute() const;
    
    // Comparison operations
    int compare(const Value& other) const;
    bool isZero() const;
    bool isNegative() const;
    bool isPositive() const;
    
    // Mathematical operations
    Value reciprocal() const;
    Value sqrt() const;
    Value reciprocalSqrt() const;
    Value scalePowerOfTwo(int power) const;
    Value exp() const;
    Value tanh() const;
    
    // Activation functions
    Value relu() const;
    
    // Utility methods
    int effectiveExponent() const {
        return static_cast<int>(exponent) - static_cast<int>(offset);
    }
    
    /*
        Width of this value's mantissa field: 6 when the sign bit was
        reused as a mantissa bit (all-positive half block, recorded by the
        AFP characterization bit), 5 otherwise. The width is a property of
        the encoded field, not inferable from the mantissa magnitude: at
        offset == 7 (no implicit leading 1) a positive-field value may
        store a raw mantissa below 32, which magnitude sniffing would
        misread as a 5-bit field and scale 2x too large.
    */
    int mantissaBits() const {
        return positive_field ? 6 : 5;
    }
    
    // Conversion helpers for internal use
    Product toProduct() const;
    static Value fromProduct(const Product& prod, int shared_exponent, bool positive_field);
    static Value fromAccumulator(const Accumulator& acc, int shared_exponent, bool positive_field);
    
    // Data members
    bool negative;
    int8_t exponent;
    uint8_t offset;
    uint8_t mantissa;
    bool positive_field;
    
private:
    static constexpr int maximum_offset = 7;
    static constexpr int fp32_exponent_bias = 127;
};

/**
 * Product represents the result of AFP multiplication in normalized form.
 * Used internally for efficient arithmetic operations.
 */
class Product {
public:
    constexpr Product() : negative(false), significand(0), scale_exponent(0), zero(true) {}
    
    constexpr Product(bool neg, uint64_t sig, int exp, bool z)
        : negative(neg), significand(sig), scale_exponent(exp), zero(z) {}
    
    bool negative;
    uint64_t significand;
    int scale_exponent;
    bool zero;
};

/**
 * Accumulator represents a high-precision accumulator for AFP arithmetic.
 * Used internally for accumulation operations with extended precision.
 */
class Accumulator {
public:
    constexpr Accumulator() : negative(false), significand(0), exponent(0), zero(true) {}
    
    constexpr Accumulator(bool neg, int64_t sig, int exp, bool z)
        : negative(neg), significand(sig), exponent(exp), zero(z) {}
    
    bool negative;
    int64_t significand;
    int exponent;
    bool zero;
};

// Utility functions for AFP arithmetic
namespace Utils {
    int integerLog2(uint64_t value);
    uint64_t shiftRightRounded(uint64_t value, int shift);
    uint64_t maskBits(int bit_count);
    int decodeSharedExponent(uint8_t encoded);
    uint8_t encodeSharedExponent(int exponent);
    void normalizeAccumulator(Accumulator& acc);
}

/**
 * Precision control for AFP arithmetic.
 *
 * Division internally scales the numerator before the integer divide;
 * the number of extra precision bits is configurable process-wide.
 * More bits yield more accurate quotients at the cost of range.
 * The default (24 bits) matches the original hard-coded value.
 */
class Precision {
public:
    static void setDivisionPrecisionBits(int bits);
    static int divisionPrecisionBits();
};

} // namespace AFP