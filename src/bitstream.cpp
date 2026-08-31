#include "../include/bitstream.hpp"

#include <stdexcept>

void BitStream::writeBits(std::uint64_t value, std::size_t bit_count)
{
    if (bit_count > 64)
    {
        throw std::invalid_argument("bit_count cannot exceed 64");
    }

    if (bit_count == 0)
    {
        return;
    }

    const std::size_t required_bits = bit_size_ + bit_count;

    const std::size_t required_bytes = (required_bits + 7) / 8;

    if (required_bytes > data_.size())
    {
        data_.resize(required_bytes, 0);
    }

    for (std::size_t i = 0; i < bit_count; ++i)
    {
        const std::size_t destination_bit = bit_size_ + i;

        const std::size_t byte_index = destination_bit / 8;

        const std::size_t bit_index = destination_bit % 8;

        const std::uint8_t bit = static_cast<std::uint8_t>((value >> i) & 1ULL);

        if (bit != 0)
        {
            data_[byte_index] |= static_cast<std::uint8_t>(1U << bit_index);
        }
        else
        {
            data_[byte_index] &= static_cast<std::uint8_t>(~(1U << bit_index));
        }
    }

    bit_size_ = required_bits;
}

std::uint64_t BitStream::readBits(std::size_t bit_offset, std::size_t bit_count) const
{
    if (bit_count > 64)
    {
        throw std::invalid_argument("bit_count cannot exceed 64");
    }

    if (bit_offset > bit_size_)
    {
        throw std::out_of_range("bit_offset is outside the bit stream");
    }

    if (bit_count > bit_size_ - bit_offset)
    {
        throw std::out_of_range("requested bits exceed the bit stream");
    }

    if (bit_count == 0)
    {
        return 0;
    }

    std::uint64_t result = 0;

    for (std::size_t i = 0; i < bit_count; ++i)
    {
        const std::size_t source_bit = bit_offset + i;

        const std::size_t byte_index = source_bit / 8;

        const std::size_t bit_index = source_bit % 8;

        const std::uint8_t bit = (data_[byte_index] >> bit_index) & 1U;

        result |= static_cast<std::uint64_t>(bit) << i;
    }

    return result;
}

std::size_t BitStream::bitSize() const
{
    return bit_size_;
}

std::size_t BitStream::byteSize() const
{
    return data_.size();
}

const std::vector<std::uint8_t>& BitStream::data() const
{
    return data_;
}

void BitStream::clear()
{
    data_.clear();
    bit_size_ = 0;
}