#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

class BitStream
{
public:
    BitStream() = default;


    void writeBits(
        std::uint64_t value,
        std::size_t bit_count
    );


    std::uint64_t readBits(
        std::size_t bit_offset,
        std::size_t bit_count
    ) const;


    std::size_t bitSize() const;


    std::size_t byteSize() const;


    const std::vector<std::uint8_t>& data() const;


    void clear();

private:
    std::vector<std::uint8_t> data_;
    std::size_t bit_size_ = 0;
};
