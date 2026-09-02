#pragma once

#include "afp.hpp"

#include <cstddef>
#include <cstdint>

class AFPArithmetic
{
public:
    static float dotProduct(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b
    );

private:
    static std::int64_t dotProductBlock(
        const AFPEncodedTensor &a,
        const AFPEncodedTensor &b,
        std::size_t block
    );

    static std::uint64_t readBits(
        const AFPEncodedTensor &tensor,
        std::size_t bit_offset,
        std::size_t bit_count
    );
};