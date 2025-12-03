#include <cstddef>
#include <cstdio>
#include <immintrin.h>

float sdot(const float *x, const float *y, const size_t n) {
    size_t mod = n % 16;
    size_t align = n - mod;
    __m512 accum = _mm512_setzero_ps();

    for (int i = 0; i <= (int)align - 16; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vy = _mm512_loadu_ps(y + i);
        accum = _mm512_fmadd_ps(vx, vy, accum);
    }

    float sum = 0.0f;
    sum = _mm512_reduce_add_ps(accum);
    for (size_t i = align; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}