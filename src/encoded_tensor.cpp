#include "../include/encoded_tensor.hpp"

std::size_t EncodedTensor::size() const
{
    return value_count_;
}

std::size_t EncodedTensor::bitSize() const
{
    return bits_.bitSize();
}

std::size_t EncodedTensor::byteSize() const
{
    return bits_.byteSize();
}

double EncodedTensor::bitsPerValue() const
{
    if (value_count_ == 0)
    {
        return 0.0;
    }

    return static_cast<double>(bitSize()) / static_cast<double>(value_count_);
}

const BitStream &EncodedTensor::bitStream() const
{
    return bits_;
}