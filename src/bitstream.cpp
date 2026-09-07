#include "../include/bitstream.hpp"

#include <stdexcept>

/*
    Bit order (unchanged from the original implementation):

    Bits are stored LSB-first within each byte and the stream grows
    from bit 0. A field written with writeBits(value, n) is read back
    with readBits(offset, n) yielding the same value.

    The fast paths below process whole bytes at a time instead of one
    bit at a time. They are bit-exact with the previous implementation:

        - writeBits clears the destination bits before setting them,
          so appending over previously-zeroed resize bytes is identical.
        - writeBits only ever appends at bit_size_, so at most the first
          touched byte can contain pre-existing data; all later bytes are
          fresh zeros and may be written whole.
        - readBits assembles up to 9 bytes LSB-first, which reproduces
          the bit-by-bit accumulation exactly (value bit i comes from
          stream bit bit_offset + i).
*/

void BitStream::loadData(
    const std::vector<std::uint8_t> &data,
    std::size_t bit_size)
{
    data_ = data;
    bit_size_ = bit_size;
}

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

    const std::size_t first_byte = bit_size_ / 8;
    const std::size_t bit_offset = bit_size_ % 8;

    /*
        Fast path: the whole field fits inside the first (partial) byte.
    */

    if (bit_offset + bit_count <= 8)
    {
        const std::uint8_t mask = static_cast<std::uint8_t>(
            ((std::uint64_t{1} << bit_count) - 1) << bit_offset);

        data_[first_byte] =
            static_cast<std::uint8_t>(
                (data_[first_byte] & static_cast<std::uint8_t>(~mask)) |
                static_cast<std::uint8_t>((value << bit_offset) & mask));

        bit_size_ = required_bits;
        return;
    }

    /*
        General case: fill the partial first byte, then write whole
        bytes, then the trailing partial byte. At most 9 bytes are
        touched because bit_count <= 64.
    */

    std::uint64_t remaining = value;
    std::size_t written = 0;
    std::size_t byte_index = first_byte;
    std::size_t bit_index = bit_offset;

    if (bit_index != 0)
    {
        const std::size_t space = 8 - bit_index;
        const std::size_t take =
            (bit_count - written < space) ? (bit_count - written) : space;

        const std::uint8_t mask = static_cast<std::uint8_t>(
            ((std::uint64_t{1} << take) - 1) << bit_index);

        data_[byte_index] =
            static_cast<std::uint8_t>(
                (data_[byte_index] & static_cast<std::uint8_t>(~mask)) |
                static_cast<std::uint8_t>(((remaining << bit_index) & mask)));

        remaining >>= take;
        written += take;
        ++byte_index;
        bit_index = 0;
    }

    while (bit_count - written >= 8)
    {
        data_[byte_index++] = static_cast<std::uint8_t>(remaining & 0xFF);
        remaining >>= 8;
        written += 8;
    }

    if (written < bit_count)
    {
        const std::size_t take = bit_count - written;

        const std::uint8_t mask = static_cast<std::uint8_t>(
            (std::uint64_t{1} << take) - 1);

        data_[byte_index] =
            static_cast<std::uint8_t>(
                (data_[byte_index] & static_cast<std::uint8_t>(~mask)) |
                static_cast<std::uint8_t>(remaining & mask));
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

    const std::size_t first_byte = bit_offset / 8;
    const std::size_t bit_index = bit_offset % 8;

    /*
        Fast path: the whole field lies within a single byte.
    */

    if (bit_index + bit_count <= 8)
    {
        const std::uint8_t mask = static_cast<std::uint8_t>(
            (std::uint64_t{1} << bit_count) - 1);

        return (data_[first_byte] >> bit_index) & mask;
    }

    /*
        General case: assemble the spanning bytes LSB-first into a
        64-bit window, then shift and mask. Fields whose bit span
        exceeds 64 bits (partial first byte + 64 field bits) are split
        into a low and a high window.
    */

    const std::size_t total_bits = bit_index + bit_count;

    if (total_bits <= 64)
    {
        std::uint64_t window = 0;

        const std::size_t span_bytes = (total_bits + 7) / 8;

        for (std::size_t i = 0; i < span_bytes; ++i)
        {
            window |=
                static_cast<std::uint64_t>(data_[first_byte + i])
                << (8 * i);
        }

        const std::uint64_t mask =
            (bit_count == 64)
                ? ~std::uint64_t{0}
                : ((std::uint64_t{1} << bit_count) - 1);

        return (window >> bit_index) & mask;
    }

    /*
        total_bits is in (64, 71]: read the low 64 - bit_index bits,
        then the remaining at most bit_index <= 7 bits.
    */

    const std::size_t low_bits = 64 - bit_index;

    std::uint64_t low_window = 0;

    for (std::size_t i = 0; i < 8; ++i)
    {
        low_window |=
            static_cast<std::uint64_t>(data_[first_byte + i])
            << (8 * i);
    }

    const std::uint64_t low = low_window >> bit_index;

    const std::size_t high_bits = bit_count - low_bits;

    std::uint64_t high_window = 0;

    const std::size_t high_bytes = (high_bits + 7) / 8;

    for (std::size_t i = 0; i < high_bytes; ++i)
    {
        high_window |=
            static_cast<std::uint64_t>(data_[first_byte + 8 + i])
            << (8 * i);
    }

    const std::uint64_t high =
        high_window & ((std::uint64_t{1} << high_bits) - 1);

    return low | (high << low_bits);
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
