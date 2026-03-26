#pragma once

// C = A * B^T + C, all row-major
// C: [M, N]
// A: [M, K]
// B: [N, K]
void matmul(const float* A, const float* B, float* C, int M, int N, int K);