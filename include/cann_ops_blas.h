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
#include <complex>
#include "cann_ops_blas_common.h"

using aclblasHandle = void *;

int aclblasScopy(float *x, float *y, const int64_t n, const int64_t incx, const int64_t incy, void *stream);

// Symmetric packed matrix-vector multiply: z = alpha * A * x + beta * y
// A is stored in packed symmetric row-major form with index: pos(i,j) = i + ((2*n - j + 1) * j) / 2
int aclblasSpmv(const float *aPacked, const float *x, const float *y, float *z,
				const float alpha, const float beta,
				const int64_t n, const int64_t incx, const int64_t incy, void *stream);

// Rank-1 matrix update: A = alpha * x * y^T + A
// A is an m��n matrix stored in row-major form with leading dimension lda
// x is a vector of length m with increment incx
// y is a vector of length n with increment incy
int aclblasSger(aclblasHandle handle, int64_t m, int64_t n, const float *alpha,
               const float *x, int64_t incx,
               float *y, int64_t incy,
               float *A, int64_t lda,
               void *stream);

// Triangular packed matrix-vector solve: solve A * x = b
// A is a triangular matrix (upper or lower), x is the solution vector
int aclblasStrsv(aclblasHandle handle,
                 aclblasFillMode uplo,
                 aclblasOperation trans,
                 aclblasDiagType diag,
                 int64_t n,
                 const float *A,
                 int64_t lda,
                 float *x,
                 int64_t incx);

int aclblasSymv(const float *a, const int64_t lda, const float *x, const float *y, float *z,
				const float alpha, const float beta,
				const int64_t n, const int64_t incx, const int64_t incy, void *stream);

int aclblasTpmv(const float *aPacked, const float *x, float *y, 
				const int64_t n, const int64_t incx, void *stream);

int aclblasTbmv(const float *a, const int64_t lda, const float *x, float *y, 
				const int64_t n, const int64_t k, const int64_t incx, void *stream);

int aclblasCdot(const float *x, const float *y, float *result,
		const int64_t n, const int64_t isConj, void *stream);

// Sum of absolute values: result = sum(|x[i]|) for i = 0 to n-1
int aclblasSasum(const float *x, float *result, const int64_t n, const int64_t incx, void *stream);

// Index of maximum absolute value: result = argmax_i |x[i]|
// Returns 1-based index following BLAS convention
// dtypeFlag: 0 for real float, 1 for complex<float>
int aclblasIamax(const float *x, int32_t *result, const int64_t n, const int64_t incx, 
                 const uint32_t dtypeFlag, void *stream);

// Column-wise complex multiplication: result[i, :] = vec[i] * mat[i, :]
// mat: m x n complex matrix (stored as 2*m*n floats)
// vec: m complex vector (stored as 2*m floats)
// result: m x n complex matrix
int aclblasColwiseMul(const float *mat, const float *vec, float *result,
                      const int64_t m, const int64_t n, void *stream);

// Euclidean norm: result = sqrt(sum(|x[i]|^2)) for i = 0 to n-1
int aclblasSnrm2(float *x, float *result, const int64_t n, const int64_t incx, void *stream);


// Complex matrix dot product: result[i,j] = matx[i,j] * maty[i,j] (element-wise complex multiplication)
// matx, maty, result: m x n complex matrices (stored as 2*m*n floats)
// Each complex number is stored as [real, imag] pairs
int aclblasComplexMatDot(const float *matx, const float *maty, float *result,
                         const int64_t m, const int64_t n, void *stream);

// Complex vector rotation: applies a plane rotation to vectors x and y
// x[i] = c*x[i] + s*y[i]
// y[i] = c*y[i] - s*x[i] (original x[i])
int aclblasCsrot(float *x, float *y, const int64_t n, const float c, const float s, void *stream);

// Batched complex matrix-vector multiplication: y[i] = alpha * A[i] * x[i] + beta * y[i]
// (or alpha * A[i]^T * x[i] + beta * y[i] or alpha * A[i]^H * x[i] + beta * y[i])
// A: batch of m x n complex matrices (batchCount matrices)
// x: batch of complex vectors (batchCount vectors)
// y: batch of complex vectors (batchCount vectors)
// alpha: complex scalar
// lda: leading dimension of matrix A
// beta: complex scalar
// incx: increment for vector x
// incy: increment for vector y
// trans: 0 = N (normal), 1 = T (transpose), 2 = C (conjugate transpose)
// dtype: 0 = half, 1 = float
int aclblasCgemvBatched(const std::complex<float> *A, const std::complex<float> *x, std::complex<float> *y,
                        const std::complex<float> &alpha, const int64_t lda,
                        const std::complex<float> &beta, const int64_t incx, const int64_t incy,
                        const int64_t batchCount, const int64_t m, const int64_t n,
                        const int32_t trans,
                        void *stream);

// Complex rank-1 update: A = alpha * x * conj(y^T) + A
// A: m x n complex matrix, data type: complex<float>
// x: complex vector of length m, data type: complex<float>
// y: complex vector of length n, data type: complex<float>
// alpha: complex scalar, data type: complex<float>
// lda: leading dimension of matrix A
// incx: increment for vector x
// incy: increment for vector y
int aclblasCgerc(const int64_t m, const int64_t n,
                 const std::complex<float> &alpha,
                 const std::complex<float> *x, const int64_t incx,
                 const std::complex<float> *y, const int64_t incy,
                 std::complex<float> *A, const int64_t lda,
                 void *stream);