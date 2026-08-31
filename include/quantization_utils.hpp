#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace quantization_utils
{
    inline int getExponent(float value)
    {
        if (value == 0.0f)
        {
            return std::numeric_limits<int>::min();
        }

        int exponent = 0;

        std::frexp(
            std::abs(value),
            &exponent
        );

        return exponent - 1;
    }


    inline int maxUnsignedValue(int bits)
    {
        if (bits <= 0)
        {
            return 0;
        }

        if (bits >= 31)
        {
            return std::numeric_limits<int>::max();
        }

        return (1 << bits) - 1;
    }


    inline std::uint64_t maxUnsignedValue64(int bits)
    {
        if (bits <= 0)
        {
            return 0;
        }

        if (bits >= 64)
        {
            return std::numeric_limits<std::uint64_t>::max();
        }

        return (std::uint64_t{1} << bits) - 1;
    }
}