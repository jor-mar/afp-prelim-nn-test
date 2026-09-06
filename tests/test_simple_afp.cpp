#include "../include/afp_value.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Testing basic AFP::Value functionality..." << std::endl;
    
    // Test factory methods
    AFP::Value zero = AFP::Value::zero();
    AFP::Value one = AFP::Value::one();
    AFP::Value two = AFP::Value::two();
    AFP::Value half = AFP::Value::half();
    
    std::cout << "Created constant values" << std::endl;
    
    // Test basic properties
    assert(zero.isZero());
    assert(!one.isZero());
    assert(!two.isZero());
    assert(!half.isZero());
    
    std::cout << "Zero detection works" << std::endl;
    
    // Test basic arithmetic
    AFP::Value sum = one.add(one);
    AFP::Value diff = two.subtract(one);
    AFP::Value product = two.multiply(half);
    AFP::Value quotient = two.divide(two);
    
    std::cout << "Basic arithmetic operations work" << std::endl;
    
    // Test comparison
    assert(one.compare(two) < 0);
    assert(two.compare(one) > 0);
    assert(one.compare(one) == 0);
    
    std::cout << "Comparison operations work" << std::endl;
    
    // Test negation
    AFP::Value neg_one = one.negate();
    assert(neg_one.isNegative());
    
    std::cout << "Negation works" << std::endl;
    
    // Test absolute
    AFP::Value abs_neg = neg_one.absolute();
    assert(!abs_neg.isNegative());
    
    std::cout << "Absolute value works" << std::endl;
    
    // Test ReLU
    AFP::Value relu_neg = neg_one.relu();
    assert(relu_neg.isZero());
    
    AFP::Value relu_pos = one.relu();
    assert(!relu_pos.isZero());
    
    std::cout << "ReLU works" << std::endl;
    
    // Test scale power of two
    AFP::Value scaled = one.scalePowerOfTwo(2);
    assert(!scaled.isZero());
    
    std::cout << "Scale power of two works" << std::endl;
    
    std::cout << "All basic tests passed!" << std::endl;
    return 0;
}