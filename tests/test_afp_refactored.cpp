#include "../include/afp_value.hpp"
#include "../include/afp_math_new.hpp"
#include "../include/afp_encoded_tensor_new.hpp"
#include <iostream>
#include <cassert>
#include <vector>

void testAFPValueOperations() {
    std::cout << "Testing AFP::Value operations..." << std::endl;
    
    // Test constant factory methods
    AFP::Value zero = AFP::Value::zero();
    AFP::Value one = AFP::Value::one();
    AFP::Value two = AFP::Value::two();
    AFP::Value half = AFP::Value::half();
    
    assert(zero.isZero());
    assert(!one.isZero());
    assert(!two.isZero());
    assert(!half.isZero());
    
    // Test basic arithmetic
    AFP::Value sum = one.add(one);
    assert(!sum.isZero());
    
    AFP::Value diff = two.subtract(one);
    assert(!diff.isZero());
    
    AFP::Value product = two.multiply(half);
    assert(!product.isZero());
    
    AFP::Value quotient = two.divide(two);
    assert(!quotient.isZero());
    
    // Test comparison
    assert(one.compare(two) < 0);
    assert(two.compare(one) > 0);
    assert(one.compare(one) == 0);
    
    // Test negation
    AFP::Value neg_one = one.negate();
    assert(neg_one.isNegative());
    
    // Test absolute
    AFP::Value abs_neg = neg_one.absolute();
    assert(!abs_neg.isNegative());
    
    // Test ReLU
    AFP::Value relu_neg = neg_one.relu();
    assert(relu_neg.isZero());
    
    AFP::Value relu_pos = one.relu();
    assert(!relu_pos.isZero());
    
    std::cout << "AFP::Value operations passed!" << std::endl;
}

void testEncodingDecoding() {
    std::cout << "Testing encoding/decoding..." << std::endl;
    
    AFPConfig config;
    AFPQuantizer quantizer(config);
    
    // Test simple encoding/decoding
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, -1.0f, -2.0f, 0.0f};
    
    AFPEncodedTensor encoded = quantizer.encode(input);
    std::vector<float> decoded = quantizer.decode(encoded);
    
    assert(decoded.size() == input.size());
    
    // Check that values are approximately preserved
    for (size_t i = 0; i < input.size(); ++i) {
        float relative_error = std::abs((decoded[i] - input[i]) / (input[i] + 1e-6f));
        assert(relative_error < 0.1f); // Allow 10% relative error due to AFP quantization
    }
    
    std::cout << "Encoding/decoding passed!" << std::endl;
}

void testTensorOperations() {
    std::cout << "Testing tensor operations..." << std::endl;
    
    AFPConfig config;
    AFPQuantizer quantizer(config);
    
    // Create test tensors
    std::vector<float> input1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> input2 = {0.5f, 1.0f, 1.5f, 2.0f};
    
    AFPEncodedTensor tensor1 = quantizer.encode(input1);
    AFPEncodedTensor tensor2 = quantizer.encode(input2);
    
    // Test element-wise operations
    AFPEncodedTensor sum = AFPArithmetic::add(tensor1, tensor2);
    AFPEncodedTensor diff = AFPArithmetic::subtract(tensor1, tensor2);
    AFPEncodedTensor product = AFPArithmetic::multiply(tensor1, tensor2);
    AFPEncodedTensor quotient = AFPArithmetic::divide(tensor1, tensor2);
    
    // Decode and verify
    std::vector<float> sum_decoded = quantizer.decode(sum);
    std::vector<float> diff_decoded = quantizer.decode(diff);
    std::vector<float> product_decoded = quantizer.decode(product);
    std::vector<float> quotient_decoded = quantizer.decode(quotient);
    
    assert(sum_decoded.size() == input1.size());
    assert(diff_decoded.size() == input1.size());
    assert(product_decoded.size() == input1.size());
    assert(quotient_decoded.size() == input1.size());
    
    // Test unary operations
    AFPEncodedTensor negated = AFPArithmetic::negate(tensor1);
    AFPEncodedTensor absolute = AFPArithmetic::absolute(tensor1);
    AFPEncodedTensor relu = AFPArithmetic::relu(tensor1);
    
    std::vector<float> negated_decoded = quantizer.decode(negated);
    std::vector<float> absolute_decoded = quantizer.decode(absolute);
    std::vector<float> relu_decoded = quantizer.decode(relu);
    
    assert(negated_decoded.size() == input1.size());
    assert(absolute_decoded.size() == input1.size());
    assert(relu_decoded.size() == input1.size());
    
    std::cout << "Tensor operations passed!" << std::endl;
}

void testReductionOperations() {
    std::cout << "Testing reduction operations..." << std::endl;
    
    AFPConfig config;
    AFPQuantizer quantizer(config);
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    AFPEncodedTensor tensor = quantizer.encode(input);
    
    // Test sum
    AFPEncodedTensor sum_result = AFPArithmetic::sum(tensor);
    std::vector<float> sum_decoded = quantizer.decode(sum_result);
    assert(sum_decoded.size() == 1);
    
    // Test mean
    AFPEncodedTensor mean_result = AFPArithmetic::mean(tensor);
    std::vector<float> mean_decoded = quantizer.decode(mean_result);
    assert(mean_decoded.size() == 1);
    
    // Test max
    AFPEncodedTensor max_result = AFPArithmetic::max(tensor);
    std::vector<float> max_decoded = quantizer.decode(max_result);
    assert(max_decoded.size() == 1);
    
    std::cout << "Reduction operations passed!" << std::endl;
}

void testMatrixOperations() {
    std::cout << "Testing matrix operations..." << std::endl;
    
    AFPConfig config;
    AFPQuantizer quantizer(config);
    
    // Test matrix-vector multiplication
    std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}; // 2x3 matrix
    std::vector<float> input = {1.0f, 2.0f, 3.0f}; // 3-element vector
    
    AFPEncodedTensor weights_tensor = quantizer.encode(weights);
    AFPEncodedTensor input_tensor = quantizer.encode(input);
    
    AFPEncodedTensor result = AFPArithmetic::matrixVectorMultiply(weights_tensor, input_tensor, 2, 3);
    std::vector<float> result_decoded = quantizer.decode(result);
    
    assert(result_decoded.size() == 2);
    
    std::cout << "Matrix operations passed!" << std::endl;
}

int main() {
    std::cout << "Running AFP refactored library tests..." << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        testAFPValueOperations();
        testEncodingDecoding();
        testTensorOperations();
        testReductionOperations();
        testMatrixOperations();
        
        std::cout << "========================================" << std::endl;
        std::cout << "All tests passed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}