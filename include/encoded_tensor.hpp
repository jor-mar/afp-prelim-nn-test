#pragma once

#include "bitstream.hpp"

#include <cstddef>
#include <string>

class EncodedTensor
{
public:
    virtual ~EncodedTensor() = default;

    std::size_t size() const;

    std::size_t bitSize() const;

    std::size_t byteSize() const;

    double bitsPerValue() const;

    virtual std::string formatName() const = 0;

    const BitStream& bitStream() const;

protected:
    BitStream bits_;
    std::size_t value_count_ = 0;
};
