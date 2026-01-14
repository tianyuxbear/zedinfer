#pragma once

#include <random>

namespace zedinfer::utils {

/**
 * @brief Access the global random engine instance (Meyers' Singleton).
 * Initializes with a non-deterministic seed by default.
 */
inline std::mt19937 &get_generator() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

// Set a fixed seed for reproducibility
inline void set_seed(int seed) {
    get_generator().seed(seed);
}

// Generate a random integer in the closed interval [a, b]
inline int randint(int a, int b) {
    std::uniform_int_distribution<int> dist(a, b);
    return dist(get_generator());
}

// Generate a random float in the interval [a, b]
inline float randfloat(float a, float b) {
    std::uniform_real_distribution<float> dist(a, b);
    return dist(get_generator());
}

} // namespace zedinfer::utils