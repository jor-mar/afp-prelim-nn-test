#include "test_utils.hpp"

namespace
{
    class TestEncodedTensor : public EncodedTensor
    {
    public:
        std::string formatName() const override
        {
            return "TEST";
        }

        void addBits(std::uint64_t value, std::size_t count)
        {
            bits_.writeBits(value, count);
        }

        void setValueCount(std::size_t count)
        {
            value_count_ = count;
        }
    };

    void testEmptyTensor()
    {
        TestEncodedTensor tensor;

        test_utils::expect(tensor.size() == 0, "Empty tensor has zero values");

        test_utils::expect(tensor.bitSize() == 0, "Empty tensor has zero bits");

        test_utils::expect(tensor.byteSize() == 0, "Empty tensor has zero bytes");

        test_utils::expectNear(tensor.bitsPerValue(), 0.0, 0.0, "Empty tensor has zero bits per value");
    }

    void testTensorMetadata()
    {
        TestEncodedTensor tensor;
        tensor.setValueCount(4);
        tensor.addBits(0xABCD, 16);

        test_utils::expect(tensor.size() == 4, "Tensor value count is correct");

        test_utils::expect(tensor.bitSize() == 16, "Tensor bit count is correct");

        test_utils::expect(tensor.byteSize() == 2, "Tensor byte count is correct");

        test_utils::expectNear(tensor.bitsPerValue(), 4.0, 1e-12, "Tensor bits per value is correct");

        test_utils::expect(tensor.formatName() == "TEST", "Tensor format name is correct");
    }

    void testBitStreamAccess()
    {
        TestEncodedTensor tensor;

        tensor.addBits(0b10101, 5);

        const BitStream &bits = tensor.bitStream();

        test_utils::expect(bits.readBits(0, 5) == 0b10101, "Bit stream access is correct");
    }

}

int main()
{
    std::cout
        << "========================================\n"
        << "EncodedTensor Tests\n"
        << "========================================\n\n";

    testEmptyTensor();
    testTensorMetadata();
    testBitStreamAccess();

    return test_utils::finish();
}
