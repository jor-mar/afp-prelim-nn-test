#pragma once
#include "afp_encoded_tensor_new.hpp"
#include "afp_value.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

class AFPArithmetic {
public:
    // Element-wise operations
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

    static AFPEncodedTensor clamp(
        const AFPEncodedTensor &input,
        const AFPEncodedTensor &min_value,
        const AFPEncodedTensor &max_value
    );

    static AFPEncodedTensor leakyRelu(
        const AFPEncodedTensor &input,
        const AFPEncodedTensor &alpha
    );

    static AFPEncodedTensor silu(
        const AFPEncodedTensor &input
    );

    // Linear algebra operations
    static AFPEncodedTensor dotProduct(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b
    );

    static AFPEncodedTensor matrixVectorMultiply(
        const AFPEncodedTensor &weights,
        const AFPEncodedTensor &input,
        std::size_t rows,
        std::size_t columns
    );

    static AFPEncodedTensor matrixMultiply(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b,
        std::size_t rows_a,
        std::size_t cols_a,
        std::size_t cols_b
    );

    // Activation functions
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

    // Reduction operations
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

    // Element-wise mathematical operations
    static AFPEncodedTensor reciprocal(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor sqrt(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor reciprocalSqrt(
        const AFPEncodedTensor &input
    );

    // Normalization
    static AFPEncodedTensor rmsNorm(
        const AFPEncodedTensor &input
    );

    static AFPEncodedTensor layerNorm(
        const AFPEncodedTensor &input
    );

    // Tensor shape operations
    static AFPEncodedTensor transpose(
        const AFPEncodedTensor &input,
        std::size_t rows,
        std::size_t columns
    );

    static AFPEncodedTensor outerProduct(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b
    );

    // Broadcasting operations
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
    // Constants
    static constexpr std::size_t block_size = 16;
    static constexpr std::size_t half_block_size = 8;
    static constexpr int maximum_offset = 7;
    static constexpr int fp32_exponent_bias = 127;

    // AFP value I/O operations
    static AFP::Value readAFPValue(
        const AFPEncodedTensor &tensor,
        std::size_t block_index,
        std::size_t value_index
    );

    static void writeAFPValue(
        BitStream &bits,
        const AFP::Value &value,
        bool positive_field
    );

    // Block-level operations
    static int computeSharedExponent(
        const std::vector<AFP::Value> &values,
        std::size_t block_start
    );

    static bool computeHalfPositive(
        const std::vector<AFP::Value> &values,
        std::size_t block_start,
        bool first_half
    );

    static uint8_t buildCharacterization(
        bool first_half_positive,
        bool second_half_positive
    );

    // Tensor construction
    static AFPEncodedTensor buildTensorFromAFPValues(
        const std::vector<AFP::Value> &values,
        const AFPConfig &config
    );

    // Utilities
    static void validateCompatible(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b
    );

    // Element-wise operation template for code reuse
    template<typename Op>
    static AFPEncodedTensor elementWiseOp(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b,
        Op operation
    ) {
        validateCompatible(a, b);
        
        std::vector<AFP::Value> output;
        output.reserve(a.size());
        
        for (std::size_t index = 0; index < a.size(); ++index) {
            const std::size_t block = index / block_size;
            const std::size_t position = index % block_size;
            
            const AFP::Value a_value = readAFPValue(a, block, position);
            const AFP::Value b_value = readAFPValue(b, block, position);
            
            output.push_back(operation(a_value, b_value));
        }
        
        return buildTensorFromAFPValues(output, a.config_);
    }

    template<typename Op>
    static AFPEncodedTensor elementWiseUnaryOp(
        const AFPEncodedTensor &input,
        Op operation
    ) {
        std::vector<AFP::Value> output;
        output.reserve(input.size());
        
        for (std::size_t index = 0; index < input.size(); ++index) {
            const std::size_t block = index / block_size;
            const std::size_t position = index % block_size;
            
            const AFP::Value value = readAFPValue(input, block, position);
            output.push_back(operation(value));
        }
        
        return buildTensorFromAFPValues(output, input.config_);
    }
};