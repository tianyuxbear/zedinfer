#pragma once

#include "utils/types.hpp"

void add_bf16(neollm::bf16_t *c, const neollm::bf16_t *a, const neollm::bf16_t *b, size_t numel);
void add_f16(neollm::fp16_t *c, const neollm::fp16_t *a, const neollm::fp16_t *b, size_t numel);
void add_f32(float *c, const float *a, const float *b, size_t numel);