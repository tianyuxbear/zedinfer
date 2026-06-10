#pragma once

#include <cstdio>
#include <iostream>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace zedinfer::utils {

inline bool streamSupportsColor(std::ostream& os) {
    if (&os == &std::cout) {
#if defined(_WIN32)
        return _isatty(_fileno(stdout)) != 0;
#else
        return isatty(fileno(stdout)) != 0;
#endif
    }
    if (&os == &std::cerr) {
#if defined(_WIN32)
        return _isatty(_fileno(stderr)) != 0;
#else
        return isatty(fileno(stderr)) != 0;
#endif
    }
    return false;
}

inline void printZedInferBanner(std::ostream& os = std::cout) {
    struct BannerLine {
        const char* zed;
        const char* infer;
    };

    static constexpr BannerLine lines[] = {
        {"░█████████                   ░██ ", "░██████               ░████"},
        {"      ░██                    ░██   ", "░██                ░██"},
        {"     ░██    ░███████   ░████████   ", "░██  ░████████  ░████████  ░███████  ░██░████"},
        {"   ░███    ░██    ░██ ░██    ░██   ", "░██  ░██    ░██    ░██    ░██    ░██ ░███"},
        {"  ░██      ░█████████ ░██    ░██   ", "░██  ░██    ░██    ░██    ░█████████ ░██"},
        {" ░██       ░██        ░██   ░███   ", "░██  ░██    ░██    ░██    ░██        ░██"},
        {"░█████████  ░███████   ░█████░██ ", "░██████░██    ░██    ░██     ░███████  ░██"},
    };

    const bool color = streamSupportsColor(os);
    const char* white = color ? "\x1B[97m" : "";
    const char* blue = color ? "\x1B[94m" : "";
    const char* reset = color ? "\x1B[0m" : "";

    os << '\n';
    for (const auto& line : lines) { os << white << line.zed << blue << line.infer << reset << '\n'; }
    os << std::endl;
}

} // namespace zedinfer::utils
