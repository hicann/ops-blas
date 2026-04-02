/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once
#include <cstdint>
#include "cann_ops_blas_common.h"

using aclblasHandle = void *;

int aclblasScopy(float *x, float *y, const int64_t n, const int64_t incx, const int64_t incy, void *stream);

// Symmetric packed matrix-vector multiply: z = alpha * A * x + beta * y
// A is stored in packed symmetric row-major form with index: pos(i,j) = i + ((2*n - j + 1) * j) / 2
int aclblasSpmv(const float *aPacked, const float *x, const float *y, float *z,
				const float alpha, const float beta,
				const int64_t n, const int64_t incx, const int64_t incy, void *stream);

// Rank-1 matrix update: A = alpha * x * y^T + A
// A is an m¡Án matrix stored in row-major form with leading dimension lda
// x is a vector of length m with increment incx
// y is a vector of length n with increment incy
int aclblasSger(aclblasHandle handle, int64_t m, int64_t n, const float *alpha,
               const float *x, int64_t incx,
               float *y, int64_t incy,
               float *A, int64_t lda,
               void *stream);