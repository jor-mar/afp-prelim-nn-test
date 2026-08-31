#include "../src/bitstream.cpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string>

namespace
{
    int failures = 0;
    void expect(bool condition, const std::string &test_name)
    {
        if (condition)
        {
            std::cout
                << "[PASS] "
                << test_name
                << '\n';
        }
        else
        {
            std::cerr
                << "[FAIL] "
                << test_name
                << '\n';

            ++failures;
        }
    }

    void expectEqual(std::uint64_t actual, std::uint64_t expected, const std::string &test_name)
    {
        expect(
            actual == expected,
            test_name);

        if (actual != expected)
        {
            std::cerr
                << "       Expected: "
                << expected
                << " (0x"
                << std::hex
                << expected
                << std::dec
                << ")\n";

            std::cerr
                << "       Actual:   "
                << actual
                << " (0x"
                << std::hex
                << actual
                << std::dec
                << ")\n";
        }
    }

    template <typename Function>
    void expectThrows(const std::string &test_name, Function&& function)
    {
        bool threw = false;

        try
        {
            function();
        }
        catch (...)
        {
            threw = true;
        }

        expect(
            threw,
            test_name);
    }

    void testEmptyStream()
    {
        BitStream stream;

        expectEqual(
            stream.bitSize(),
            0,
            "Empty stream has zero bits");

        expectEqual(
            stream.byteSize(),
            0,
            "Empty stream has zero bytes");

        expect(
            stream.data().empty(),
            "Empty stream has empty data");
    }

    void testSingleBit()
    {
        BitStream stream;

        stream.writeBits(1, 1);

        expectEqual(
            stream.bitSize(),
            1,
            "Single bit increases bit size to 1");

        expectEqual(
            stream.byteSize(),
            1,
            "Single bit requires one byte");

        expectEqual(
            stream.readBits(0, 1),
            1,
            "Single bit reads correctly");
    }

    void testIndividualBitValues()
    {
        BitStream stream;

        stream.writeBits(0, 1);
        stream.writeBits(1, 1);
        stream.writeBits(0, 1);
        stream.writeBits(1, 1);

        expectEqual(
            stream.bitSize(),
            4,
            "Four individual bits produce four bits");

        expectEqual(
            stream.readBits(0, 1),
            0,
            "First individual bit is correct");

        expectEqual(
            stream.readBits(1, 1),
            1,
            "Second individual bit is correct");

        expectEqual(
            stream.readBits(2, 1),
            0,
            "Third individual bit is correct");

        expectEqual(
            stream.readBits(3, 1),
            1,
            "Fourth individual bit is correct");

        expectEqual(
            stream.readBits(0, 4),
            0b1010,
            "Combined individual bits are correct");
    }

    void testBasicWrites()
    {
        BitStream stream;

        stream.writeBits(0b1011, 4);
        stream.writeBits(0b110, 3);
        stream.writeBits(0b10010, 5);

        expectEqual(
            stream.bitSize(),
            12,
            "Multiple writes produce correct total bit size");

        expectEqual(
            stream.readBits(0, 4),
            0b1011,
            "First field reads correctly");

        expectEqual(
            stream.readBits(4, 3),
            0b110,
            "Second field reads correctly");

        expectEqual(
            stream.readBits(7, 5),
            0b10010,
            "Third field reads correctly");
    }

    void testByteBoundary()
    {
        BitStream stream;

        stream.writeBits(0b10101010, 8);
        stream.writeBits(0b11001100, 8);

        expectEqual(
            stream.bitSize(),
            16,
            "Two bytes produce 16 bits");

        expectEqual(
            stream.byteSize(),
            2,
            "Two bytes require two bytes");

        expectEqual(
            stream.readBits(0, 8),
            0b10101010,
            "First byte reads correctly");

        expectEqual(
            stream.readBits(8, 8),
            0b11001100,
            "Second byte reads correctly");
    }

    void testCrossByteBoundary()
    {
        BitStream stream;

        stream.writeBits(0b101, 3);
        stream.writeBits(0b11010101, 8);
        stream.writeBits(0b111, 3);

        expectEqual(
            stream.bitSize(),
            14,
            "Cross-byte fields produce correct bit size");

        expectEqual(
            stream.readBits(0, 3),
            0b101,
            "Field before byte boundary reads correctly");

        expectEqual(
            stream.readBits(3, 8),
            0b11010101,
            "Field crossing byte boundary reads correctly");

        expectEqual(
            stream.readBits(11, 3),
            0b111,
            "Field after byte boundary reads correctly");
    }

    void testNineBitFields()
    {
        /*
            AFP values will eventually contain:

                1 sign bit
                3 offset bits
                5 mantissa bits

            = 9 bits per value

            This test specifically verifies consecutive
            9-bit fields.
        */

        BitStream stream;

        const std::uint64_t values[] = {
            0b101010101,
            0b010101010,
            0b111000111,
            0b000111000,
            0b111111111};

        for (const auto value : values)
        {
            stream.writeBits(value, 9);
        }

        expectEqual(
            stream.bitSize(),
            45,
            "Five 9-bit fields produce 45 bits");

        for (std::size_t i = 0; i < 5; ++i)
        {
            expectEqual(
                stream.readBits(i * 9, 9),
                values[i],
                "9-bit field " +
                    std::to_string(i) +
                    " reads correctly");
        }
    }

    void testLargeFields()
    {
        BitStream stream;

        const std::uint64_t values[] = {
            0xFF,
            0xFFFF,
            0xFFFFFFFF,
            0xFFFFFFFFFFFFFFFFULL};

        const std::size_t widths[] = {
            8,
            16,
            32,
            64};

        std::size_t offset = 0;

        for (std::size_t i = 0; i < 4; ++i)
        {
            stream.writeBits(
                values[i],
                widths[i]);

            expectEqual(
                stream.readBits(
                    offset,
                    widths[i]),
                values[i],
                std::to_string(widths[i]) +
                    "-bit field reads correctly");

            offset += widths[i];
        }

        expectEqual(
            stream.bitSize(),
            120,
            "Large fields produce correct total bit size");
    }

    void testZeroBitWrite()
    {
        BitStream stream;

        stream.writeBits(
            0xFFFFFFFFFFFFFFFFULL,
            0);

        expectEqual(
            stream.bitSize(),
            0,
            "Writing zero bits does not change stream size");

        expectEqual(
            stream.byteSize(),
            0,
            "Writing zero bits does not allocate storage");
    }

    void testZeroBitRead()
    {
        BitStream stream;

        stream.writeBits(
            0b10101010,
            8);

        expectEqual(
            stream.readBits(4, 0),
            0,
            "Reading zero bits returns zero");
    }

    void testTruncation()
    {
        BitStream stream;

        stream.writeBits(
            0b1111,
            4);

        /*
            Only the lowest four bits should be stored.
        */

        expectEqual(
            stream.readBits(0, 4),
            0b1111,
            "Write stores only requested number of bits");

        BitStream second;

        second.writeBits(
            0b11110000,
            4);

        expectEqual(
            second.readBits(0, 4),
            0b0000,
            "Higher bits are ignored during write");
    }

    void testClear()
    {
        BitStream stream;

        stream.writeBits(
            0xFFFF,
            16);

        stream.clear();

        expectEqual(
            stream.bitSize(),
            0,
            "Clear resets bit size");

        expectEqual(
            stream.byteSize(),
            0,
            "Clear releases byte storage");

        expect(
            stream.data().empty(),
            "Clear empties underlying data");
    }

    void testRepeatedClear()
    {
        BitStream stream;

        stream.writeBits(0xFF, 8);

        stream.clear();
        stream.clear();

        expectEqual(
            stream.bitSize(),
            0,
            "Repeated clear leaves stream empty");
    }

    void testInvalidWrite()
    {
        BitStream stream;

        expectThrows(
            "Writing more than 64 bits throws",
            [&]()
            {
                stream.writeBits(
                    0,
                    65);
            });
    }

    void testInvalidRead()
    {
        BitStream stream;

        stream.writeBits(
            0xFF,
            8);

        expectThrows(
            "Reading more than 64 bits throws",
            [&]()
            {
                stream.readBits(
                    0,
                    65);
            });

        expectThrows(
            "Reading beyond stream throws",
            [&]()
            {
                stream.readBits(
                    8,
                    1);
            });

        expectThrows(
            "Reading from invalid offset throws",
            [&]()
            {
                stream.readBits(
                    9,
                    1);
            });

        expectThrows(
            "Reading past stream end throws",
            [&]()
            {
                stream.readBits(
                    4,
                    5);
            });
    }

    void testDataStorage()
    {
        BitStream stream;

        stream.writeBits(
            0xAB,
            8);

        stream.writeBits(
            0xCD,
            8);

        const auto &data = stream.data();

        expectEqual(
            data.size(),
            2,
            "Underlying data has correct byte count");

        /*
            Bits are stored least-significant-bit first
            within each byte, so the byte values themselves
            should still reconstruct to the original values.
        */

        expectEqual(
            data[0],
            0xAB,
            "First stored byte is correct");

        expectEqual(
            data[1],
            0xCD,
            "Second stored byte is correct");
    }

    void testUnalignedStorage()
    {
        BitStream stream;

        /*
            Create 3 + 5 + 7 + 11 = 26 bits.
        */

        stream.writeBits(0b101, 3);
        stream.writeBits(0b11010, 5);
        stream.writeBits(0b1010101, 7);
        stream.writeBits(0b10101010101, 11);

        expectEqual(
            stream.bitSize(),
            26,
            "Unaligned fields produce correct total size");

        expectEqual(
            stream.readBits(0, 3),
            0b101,
            "First unaligned field reads correctly");

        expectEqual(
            stream.readBits(3, 5),
            0b11010,
            "Second unaligned field reads correctly");

        expectEqual(
            stream.readBits(8, 7),
            0b1010101,
            "Third unaligned field reads correctly");

        expectEqual(
            stream.readBits(15, 11),
            0b10101010101,
            "Fourth unaligned field reads correctly");
    }

}

int main()
{
    std::cout
        << "BitStream Tests\n";

    testEmptyStream();

    testSingleBit();

    testIndividualBitValues();

    testBasicWrites();

    testByteBoundary();

    testCrossByteBoundary();

    testNineBitFields();

    testLargeFields();

    testZeroBitWrite();

    testZeroBitRead();

    testTruncation();

    testClear();

    testRepeatedClear();

    testInvalidWrite();

    testInvalidRead();

    testDataStorage();

    testUnalignedStorage();

    std::cout
        << "\n========================================\n";

    if (failures == 0)
    {
        std::cout
            << "ALL TESTS PASSED\n";

        return 0;
    }

    std::cerr
        << failures
        << " TEST(S) FAILED\n";

    return 1;
}
