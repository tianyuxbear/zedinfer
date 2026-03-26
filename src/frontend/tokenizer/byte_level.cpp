#include "frontend/tokenizer/byte_level.hpp"
#include "utils/logging.hpp"

#include <algorithm>
#include <plog/Log.h>
#include <vector>

namespace zedinfer::tokenizer {

// Static member definitions.
const std::unordered_map<unsigned char, std::string> ByteLevel::byte_to_unicode_map_
    = ByteLevel::create_byte_to_unicode_map();

const std::unordered_map<UChar32, unsigned char> ByteLevel::unicode_to_byte_map_
    = ByteLevel::create_unicode_to_byte_map();

std::unordered_map<unsigned char, std::string> ByteLevel::create_byte_to_unicode_map() {
    std::vector<int> bytes;
    // Printable ASCII: '!' to '~' (33–126)
    for (int b = 33; b <= 126; ++b) { bytes.push_back(b); }
    // Extended ASCII: ¡–¬ (161–172) and ®–ÿ (174–255)
    for (int b = 161; b <= 172; ++b) { bytes.push_back(b); }
    for (int b = 174; b <= 255; ++b) { bytes.push_back(b); }

    std::vector<int> unicode_chars = bytes;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bytes.begin(), bytes.end(), b) == bytes.end()) {
            bytes.push_back(b);
            unicode_chars.push_back(256 + n);
            ++n;
        }
    }

    std::unordered_map<unsigned char, std::string> map;
    for (size_t i = 0; i < bytes.size(); ++i) {
        icu::UnicodeString uStr(static_cast<UChar32>(unicode_chars[i]));
        std::string utf8;
        uStr.toUTF8String(utf8);
        map[static_cast<unsigned char>(bytes[i])] = utf8;
    }
    return map;
}

std::unordered_map<UChar32, unsigned char> ByteLevel::create_unicode_to_byte_map() {
    std::vector<int> bytes;
    for (int b = 33; b <= 126; ++b) { bytes.push_back(b); }
    for (int b = 161; b <= 172; ++b) { bytes.push_back(b); }
    for (int b = 174; b <= 255; ++b) { bytes.push_back(b); }

    std::vector<int> unicode_chars = bytes;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bytes.begin(), bytes.end(), b) == bytes.end()) {
            bytes.push_back(b);
            unicode_chars.push_back(256 + n++);
        }
    }

    std::unordered_map<UChar32, unsigned char> map;
    for (size_t i = 0; i < bytes.size(); ++i) {
        map[static_cast<UChar32>(unicode_chars[i])] = static_cast<unsigned char>(bytes[i]);
    }
    return map;
}

std::string ByteLevel::byte_to_unicode(unsigned char byte) {
    auto it = byte_to_unicode_map_.find(byte);
    if (it != byte_to_unicode_map_.end()) {
        return it->second;
    }
    // Should never happen: all 256 bytes are mapped.
    LOG_ERROR_(utils::BOTH) << "[ByteLevel] Warning: No mapping for byte " << static_cast<int>(byte) << '\n';
    return std::string(1, static_cast<char>(byte));
}

unsigned char ByteLevel::unicode_to_byte(UChar32 unicode_char) {
    auto it = unicode_to_byte_map_.find(unicode_char);
    if (it != unicode_to_byte_map_.end()) {
        return it->second;
    }
    // Fallback for raw bytes in 0–255 range.
    if (unicode_char <= 255) {
        return static_cast<unsigned char>(unicode_char);
    }
    LOG_ERROR_(utils::BOTH) << "[ByteLevel] Warning: Unknown Unicode char: " << unicode_char << '\n';
    return 0;
}

std::string ByteLevel::bytes_to_unicode(const std::string& text) {
    std::string result;
    result.reserve(text.size() * 3); // UTF-8 chars may be multi-byte.
    for (unsigned char byte : text) { result += byte_to_unicode(byte); }
    return result;
}

std::string ByteLevel::unicode_to_bytes(const std::string& text) {
    std::string result;
    icu::UnicodeString uText = icu::UnicodeString::fromUTF8(text);
    for (int32_t i = 0; i < uText.length(); ++i) {
        UChar32 code_point = uText.char32At(i);
        unsigned char byte = unicode_to_byte(code_point);
        result += static_cast<char>(byte);
        if (code_point > 0xFFFF) {
            ++i; // Skip low surrogate.
        }
    }
    return result;
}

} // namespace zedinfer::tokenizer