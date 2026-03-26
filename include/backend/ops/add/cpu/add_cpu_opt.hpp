#pragma once

#include "utils/types.hpp"

void add_bf16(zedinfer::bf16_t* c, const zedinfer::bf16_t* a, const zedinfer::bf16_t* b, size_t numel);
void add_f16(zedinfer::fp16_t* c, const zedinfer::fp16_t* a, const zedinfer::fp16_t* b, size_t numel);
void add_f32(float* c, const float* a, const float* b, size_t numel);