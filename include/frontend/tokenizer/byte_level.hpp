#pragma once

#include <string>
#include <unicode/unistr.h>
#include <unordered_map>

namespace zedinfer::tokenizer {

/**
 * Byte-level encoder/decoder (GPT-2 style).
 * Maps arbitrary bytes to printable Unicode characters for reversible, UTF-8-safe BPE.
 */
class ByteLevel {
public:
    // Map a single byte to its Unicode string representation.
    static std::string byte_to_unicode(unsigned char byte);

    // Map a Unicode code point back to its original byte.
    static unsigned char unicode_to_byte(UChar32 unicode_char);

    // Encode byte sequence to Unicode string.
    static std::string bytes_to_unicode(const std::string& text);

    // Decode Unicode string back to original bytes.
    static std::string unicode_to_bytes(const std::string& text);

private:
    // Build static lookup tables.
    static std::unordered_map<unsigned char, std::string> create_byte_to_unicode_map();
    static std::unordered_map<UChar32, unsigned char> create_unicode_to_byte_map();

    // Precomputed bidirectional mappings.
    static const std::unordered_map<unsigned char, std::string> byte_to_unicode_map_;
    static const std::unordered_map<UChar32, unsigned char> unicode_to_byte_map_;
};

} // namespace zedinfer::tokenizer