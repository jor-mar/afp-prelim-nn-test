#pragma once
#include "afp_encoded_tensor_new.hpp"
#include "afp_value.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

/*
    A dense matrix decoded once from its AFP bitstream at model-load
    time. Each row holds one AFP::Product per weight: the exact integer
    significand/scale pair that toProduct() would return, so kernels
    using it are bit-exact with the decode-every-call implementations.

    The pointed-to representation is an implementation detail; copy
    semantics are disabled because rows alias the heap buffer.
*/
class AFPPreparedMatrix {
public:
    AFPPreparedMatrix();
    ~AFPPreparedMatrix();

    AFPPreparedMatrix(const AFPPreparedMatrix &) = delete;
    AFPPreparedMatrix &operator=(const AFPPreparedMatrix &) = delete;
    AFPPreparedMatrix(AFPPreparedMatrix &&) noexcept;
    AFPPreparedMatrix &operator=(AFPPreparedMatrix &&) noexcept;

    std::size_t rows() const;
    std::size_t columns() const;

    /*
        Row-major access: entry (row, column) is the AFP::Product for
        that weight, ready for the multiply-accumulate kernel.
    */
    const AFP::Product &product(std::size_t row, std::size_t column) const;

private:
    friend class AFPArithmetic;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/*
    AFP-NATIVE ARITHMETIC CONTRACT
    ------------------------------
    Every operation in AFPArithmetic is computed entirely in AFP
    representation: tensors are decoded into AFP::Value, arithmetic runs
    on the integer significand/scale-exponent form (Product/Accumulator),
    and results are re-encoded with buildTensorFromAFPValues.

    No operation converts AFP data to float/double to perform arithmetic
    and back. The only sanctioned float conversions in the library are
    the boundary I/O in AFPQuantizer::encode / AFPQuantizer::decode.
*/

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

    // ------------------------------------------------------------------
    // Prepared-weight and unpacked-value inference API
    //
    // prepareMatrix decodes a weight tensor once (call at model load).
    //
    // Static model data (weights) and intermediate activations pay a
    // heavy decode/encode toll when every operation works on the packed
    // 9-bit bitstream. The API below keeps the AFP-native contract
    // (integer significand/scale-exponent arithmetic only) while moving
    // the conversion cost to where it is paid once:
    //
    //   - AFPPreparedMatrix decodes a weight tensor exactly once at
    //     model load and stores each row as AFP::Product entries.
    //   - matrixVectorMultiplyPrepared runs the same accumulation as
    //     matrixVectorMultiply (bit-exact, same rounding) against the
    //     prepared rows.
    //   - addUnpacked / reluUnpacked operate on plain AFP::Value
    //     vectors, and encodeAFPValues packs them back to a tensor.
    //
    // Encoding only at I/O boundaries removes the repeated
    // Value -> Product -> Accumulator -> Value churn between layers
    // without ever converting to float or double.
    // ------------------------------------------------------------------

    static std::unique_ptr<AFPPreparedMatrix> prepareMatrix(
        const AFPEncodedTensor &weights,
        std::size_t rows,
        std::size_t columns
    );

    static AFPEncodedTensor matrixVectorMultiplyPrepared(
        const AFPPreparedMatrix &prepared,
        const AFPEncodedTensor &input
    );

    static std::vector<AFP::Value> matrixVectorMultiplyUnpacked(
        const AFPPreparedMatrix &prepared,
        const std::vector<AFP::Value> &input
    );

    static std::vector<AFP::Value> addUnpacked(
        const std::vector<AFP::Value> &a,
        const std::vector<AFP::Value> &b
    );

    static std::vector<AFP::Value> reluUnpacked(
        const std::vector<AFP::Value> &input
    );

    static AFPEncodedTensor encodeAFPValues(
        const std::vector<AFP::Value> &values,
        const AFPConfig &config
    );

    static std::vector<AFP::Value> decodeAFPValues(
        const AFPEncodedTensor &tensor
    );

private:
    // Shared kernel behind matrixVectorMultiplyPrepared / matrixVectorMultiplyUnpacked
    static void preparedMatVecKernel(
        const AFPPreparedMatrix &prepared,
        const AFP::Product *input_products,
        AFP::Value *output
    );

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

    // Bulk block decoding (avoids per-element header re-parsing)
    static void decodeBlock(
        const AFPEncodedTensor &tensor,
        std::size_t block_index,
        AFP::Value *out_values
    );

    // Decode a contiguous run of values (row-major, cache-friendly order)
    static void decodeRow(
        const AFPEncodedTensor &tensor,
        std::size_t start_value,
        std::size_t count,
        AFP::Value *out_values
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

    // Accumulate a product into a running AFP accumulator
    static void accumulateProduct(
        AFP::Accumulator &accumulator,
        const AFP::Product &product
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
        
        /*
            Decode each block of both operands once instead of once per
            element position (common subexpression elimination): block
            headers are no longer re-parsed for every element, and the
            two operands are traversed in lockstep for cache locality.
        */
        
        AFP::Value a_block[block_size];
        AFP::Value b_block[block_size];
        
        const std::size_t element_count = a.size();
        const std::size_t full_blocks = element_count / block_size;
        
        for (std::size_t block = 0; block < full_blocks; ++block) {
            decodeBlock(a, block, a_block);
            decodeBlock(b, block, b_block);
            
            for (std::size_t position = 0; position < block_size; ++position) {
                output.push_back(operation(a_block[position], b_block[position]));
            }
        }
        
        const std::size_t remainder = element_count % block_size;
        
        if (remainder != 0) {
            decodeBlock(a, full_blocks, a_block);
            decodeBlock(b, full_blocks, b_block);
            
            for (std::size_t position = 0; position < remainder; ++position) {
                output.push_back(operation(a_block[position], b_block[position]));
            }
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
        
        /*
            Block-at-a-time decoding: shared block header bits are read
            once per block instead of once per element.
        */
        
        AFP::Value input_block[block_size];
        
        const std::size_t element_count = input.size();
        const std::size_t full_blocks = element_count / block_size;
        
        for (std::size_t block = 0; block < full_blocks; ++block) {
            decodeBlock(input, block, input_block);
            
            for (std::size_t position = 0; position < block_size; ++position) {
                output.push_back(operation(input_block[position]));
            }
        }
        
        const std::size_t remainder = element_count % block_size;
        
        if (remainder != 0) {
            decodeBlock(input, full_blocks, input_block);
            
            for (std::size_t position = 0; position < remainder; ++position) {
                output.push_back(operation(input_block[position]));
            }
        }
        
        return buildTensorFromAFPValues(output, input.config_);
    }
};