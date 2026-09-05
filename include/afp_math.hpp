#pragma once
#include "afp_encoded_tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

class AFPArithmetic
{
public:
    /*
     * Element-wise operations.
     *
     * All input and output tensors remain encoded in AFP format.
     */
    static AFPEncodedTensor add(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b
    );

    static AFPEncodedTensor subtract(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b
    );

    static AFPEncodedTensor multiply(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b
    );

    static AFPEncodedTensor divide(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b
    );

    static AFPEncodedTensor negate(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor absolute(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor minimum(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b
    );

    static AFPEncodedTensor maximum(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b
    );

    /*
     * Clamp every value into [min_value, max_value].
     *
     * min_value and max_value must each contain exactly one AFP value.
     */
    static AFPEncodedTensor clamp(
        const AFPEncodedTensor &input,
        const AFPEncodedTensor &min_value,
        const AFPEncodedTensor &max_value
    );

    /*
     * Leaky ReLU.
     *
     * negative values are multiplied by alpha.
     * alpha must contain exactly one AFP value.
     */
    static AFPEncodedTensor leakyRelu(
        const AFPEncodedTensor &input,
        const AFPEncodedTensor &alpha
    );

    /*
     * SiLU(x) = x * sigmoid(x)
     */
    static AFPEncodedTensor silu(
        const AFPEncodedTensor &input
    );

    /*
     * Dot product.
     */
    static AFPEncodedTensor dotProduct(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b
    );

    /*
     * Matrix-vector multiplication.
     */
    static AFPEncodedTensor matrixVectorMultiply(
        const AFPEncodedTensor &weights,
        const AFPEncodedTensor &input,
        std::size_t rows,
        std::size_t columns
    );

    /*
     * Matrix-matrix multiplication.
     */
    static AFPEncodedTensor matrixMultiply(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b,
        std::size_t rows_a,
        std::size_t cols_a,
        std::size_t cols_b
    );

    /*
     * Activation functions.
     */
    static AFPEncodedTensor relu(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor sigmoid(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor tanh(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor gelu(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor exp(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor softmax(
        const AFPEncodedTensor &input
    );

    /*
     * Reduction operations.
     */
    static AFPEncodedTensor sum(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor mean(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor max(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor sumSquares(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor rms(
        const AFPEncodedTensor &input
    );

    /*
     * Element-wise mathematical operations.
     */
    static AFPEncodedTensor reciprocal(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor sqrt(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor reciprocalSqrt(
        const AFPEncodedTensor &input
    );

    /*
     * Normalization.
     */
    static AFPEncodedTensor rmsNorm(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor layerNorm(
        const AFPEncodedTensor &input
    );

    /*
     * Tensor shape operations.
     *
     * Input is row-major.
     */
    static AFPEncodedTensor transpose(
        const AFPEncodedTensor &input,
        std::size_t rows,
        std::size_t columns
    );

    /*
     * Outer product.
     *
     * a has m values.
     * b has n values.
     * result has m*n values in row-major order.
     */
    static AFPEncodedTensor outerProduct(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b
    );

    /*
     * Broadcasting operations.
     */
    static AFPEncodedTensor broadcastAdd(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b,
        std::size_t target_size
    );

    static AFPEncodedTensor broadcastMultiply(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b,
        std::size_t target_size
    );

private:
    /*
    struct AFPValue
    {
        bool negative = false;
        int exponent = -126;
        int offset = 7;
        std::uint64_t mantissa = 0;
        int mantissa_bits = 5;
        bool zero = true;
    };
    */
    struct AFPValue
    {
        std::int8_t exponent = -126;
        std::uint8_t offset = 7;
        std::uint8_t mantissa = 0;
        bool negative = false;
    };


    struct AFPProduct
    {
        bool negative = false;
        std::uint64_t significand = 0;
        int scale_exponent = 0;
        bool zero = true;
    };

    struct AFPAccumulator
    {
        bool negative = false;
        std::int64_t significand = 0;
        int exponent = 0;
        bool zero = true;
    };

    static constexpr std::size_t block_size = 16;
    static constexpr std::size_t half_block_size = 8;
    static constexpr int maximum_offset = 7;
    static constexpr int fp32_exponent_bias = 127;

    /*
     * AFP field operations.
     */
    static AFPValue readAFPValue(
        const AFPEncodedTensor &tensor,
        std::size_t block_index,
        std::size_t value_index
    );

    static AFPProduct normalizeToProduct(
        const AFPValue &value
    );

    static AFPProduct multiplyAFPValues(
        const AFPValue &a,
        const AFPValue &b
    );

    static AFPProduct divideAFPValues(
        const AFPValue &a,
        const AFPValue &b
    );

    static AFPAccumulator addProducts(
        const AFPAccumulator &acc,
        const AFPProduct &prod
    );

    static AFPAccumulator addAFPValues(
        const AFPValue &a,
        const AFPValue &b
    );

    static AFPAccumulator subtractAFPValues(
        const AFPValue &a,
        const AFPValue &b
    );

    static AFPValue accumulatorToAFPValue(
        const AFPAccumulator &acc,
        int shared_exponent,
        bool positive_field
    );

    static AFPValue productToAFPValue(
        const AFPProduct &prod,
        int shared_exponent,
        bool positive_field
    );

    /*
     * Comparisons and activations.
     */
    static bool isAFPValueZero(
        const AFPValue &value
    );

    static bool isAFPValueNegative(
        const AFPValue &value
    );

    static int compareAFPValues(
        const AFPValue &a,
        const AFPValue &b
    );

    static AFPValue applyReLU(
        const AFPValue &value
    );

    /*
     * Special-value constructors.
     */
    static AFPValue zeroValue();

    static AFPValue oneValue();

    static AFPValue twoValue();

    static AFPValue halfValue();

    /*
     * Scalar arithmetic helpers.
     */
    static AFPValue addValues(
        const AFPValue &a,
        const AFPValue &b
    );

    static AFPValue subtractValues(
        const AFPValue &a,
        const AFPValue &b
    );

    static AFPValue multiplyValues(
        const AFPValue &a,
        const AFPValue &b
    );

    static AFPValue divideValues(
        const AFPValue &a,
        const AFPValue &b
    );

    static AFPValue scalePowerOfTwo(
        const AFPValue &value,
        int power
    );

    static AFPValue reciprocalValue(
        const AFPValue &value
    );

    static AFPValue sqrtValue(
        const AFPValue &value
    );

    static AFPValue expValue(
        const AFPValue &value
    );

    static AFPValue tanhValue(
        const AFPValue &value
    );

    /*
     * Block-level operations.
     */
    static int computeSharedExponent(
        const std::vector<AFPValue> &values,
        std::size_t block_start
    );

    static bool computeHalfPositive(
        const std::vector<AFPValue> &values,
        std::size_t block_start,
        bool first_half
    );

    static std::uint8_t buildCharacterization(
        bool first_half_positive,
        bool second_half_positive
    );

    /*
     * Tensor construction.
     */
    static AFPEncodedTensor buildTensorFromAFPValues(
        const std::vector<AFPValue> &values,
        const AFPConfig &config
    );

    static void writeAFPValue(
        BitStream &bits,
        const AFPValue &value,
        bool positive_field
    );

    /*
     * Utilities.
     */
    static int integerLog2(
        std::uint64_t value
    );

    static std::uint64_t shiftRightRounded(
        std::uint64_t value,
        int shift
    );

    static std::uint64_t maskBits(
        int bit_count
    );

    static int decodeSharedExponent(
        std::uint8_t encoded
    );

    static std::uint8_t encodeSharedExponent(
        int exponent
    );

    static void validateCompatible(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b
    );

    static void normalizeAccumulator(
        AFPAccumulator &acc
    );

    static bool isZeroAFPValue(const AFPValue &value);

    static int mantissaBitsForValue(const AFPValue &value);
};