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
#include <cstddef>
#include <acl/acl.h>
#include "cann_ops_blas_common.h"

using aclblasHandle = void *;
using aclblasLogCallback = void (*)(char*);

typedef void* aclblasHandle_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 创建 ops-blas handle
 *
 * 在堆上分配一个 ops-blas handle，并通过 @p handle 输出给调用者。内部结构
 * 对用户不可见，用户通过返回的 void* 使用其他 API。
 *
 * @param handle 输出参数，用于接收创建的 handle 指针（void**）。调用前 *handle
 *               必须为 nullptr，否则视为已存在以防止内存泄漏。
 * @return ACLBLAS_STATUS_SUCCESS 成功
 *         ACLBLAS_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 *         ACLBLAS_STATUS_INVALID_VALUE *handle 非空（防止重复创建导致内存泄漏）
 *         ACLBLAS_STATUS_NOT_INITIALIZED CANN 上下文未初始化
 *         ACLBLAS_STATUS_ALLOC_FAILED 内存分配失败
 */
aclblasStatus_t aclblasCreate(aclblasHandle_t *handle);

/**
 * @brief 销毁 ops-blas handle
 *
 * 释放由 aclblasCreate 创建的 handle 所占有的全部资源。
 *
 * @param handle 要销毁的 handle（void*）
 * @return ACLBLAS_STATUS_SUCCESS 成功
 *         ACLBLAS_STATUS_HANDLE_IS_NULLPTR handle 为空
 */
aclblasStatus_t aclblasDestroy(aclblasHandle_t handle);

/**
 * @brief 设置 handle 的 stream
 * @param handle aclblas handle（void*）
 * @param stream 要设置的 stream，nullptr 表示使用默认 stream
 * @return ACLBLAS_STATUS_SUCCESS 成功
 *         ACLBLAS_STATUS_HANDLE_IS_NULLPTR handle 为空
 */
aclblasStatus_t aclblasSetStream(aclblasHandle_t handle, aclrtStream stream);

/**
 * @brief 获取 handle 的 stream
 * @param handle aclblas handle（void*）
 * @param stream 输出参数，返回当前 stream
 * @return ACLBLAS_STATUS_SUCCESS 成功
 *         ACLBLAS_STATUS_HANDLE_IS_NULLPTR handle 为空
 *         ACLBLAS_STATUS_INVALID_VALUE stream 输出参数为空
 */
aclblasStatus_t aclblasGetStream(aclblasHandle_t handle, aclrtStream *stream);

/**
 * @brief 设置 handle 的 workspace
 *
 * 供算子内部临时存储使用。允许用户自行管理 workspace 内存；传入
 * (nullptr, 0) 可以清空 handle 中的 workspace 设置。
 *
 * @param handle aclblas handle（void*）
 * @param workspace 用户分配的 device 内存指针
 * @param workspaceSize workspace 大小（字节）
 * @return ACLBLAS_STATUS_SUCCESS 成功
 *         ACLBLAS_STATUS_HANDLE_IS_NULLPTR handle 为空
 *         ACLBLAS_STATUS_INVALID_VALUE workspace 与 workspaceSize 语义冲突
 */
aclblasStatus_t aclblasSetWorkspace(aclblasHandle_t handle, void *workspace, size_t workspaceSize);

/**
 * @brief 获取 ops-blas 版本号
 *
 * 版本号编码为 MAJOR * 10000 + MINOR * 100 + PATCH，如 10000 表示 1.0.0。
 *
 * @param handle  handle指针，可为NULL
 * @param version 输出参数，接收版本号
 * @return ACLBLAS_STATUS_SUCCESS 成功
 *         ACLBLAS_STATUS_INVALID_VALUE 参数无效
 */
aclblasStatus_t aclblasGetVersion(aclblasHandle_t handle, int *version);

/**
 * @brief 配置日志接口
 * @param logFile  日志文件名称，为空不输出
 * @param logToStdOut 是否输出到标准流
 * @param logToKdlls  是否输出到内核
 * @param logLevel 日志级别
 * @return ACLBLAS_STATUS_SUCCESS 成功
 *         ACLBLAS_STATUS_INVALID_VALUE 参数无效
 */
aclblasStatus_t aclblasLoggerConfigure(const char* logFile, bool logToStdOut, bool logToKdlls, aclblasLogLevel_t logLevel);

/**
 * @brief 设置日志回调函数
 * @param handle  句柄
 * @param userCallback 日志回调函数
 * @return ACLBLAS_STATUS_SUCCESS 成功
 *         ACLBLAS_STATUS_HANDLE_IS_NULLPTR handle 为空
 */
aclblasStatus_t aclblasSetLoggerCallback(aclblasHandle handle, aclblasLogCallback userCallback);

/**
 * @brief 获取日志回调函数
 * @param handle  句柄
 * @param userCallback 日志回调函数
 * @return ACLBLAS_STATUS_SUCCESS 成功
 *         ACLBLAS_STATUS_HANDLE_IS_NULLPTR handle 为空
 */
aclblasStatus_t aclblasGetLoggerCallback(aclblasHandle handle, aclblasLogCallback userCallback);

#ifdef __cplusplus
}
#endif

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

int aclblasCcopy(std::complex<float> *x, std::complex<float> *y, const int64_t n, const int64_t incx, const int64_t incy, void *stream);


// Real vector dot product: result = sum(x[i] * y[i]) for i = 0 to n-1
int aclblasSdot(aclblasHandle handle, const float *x, const float *y, float *result,
                const int64_t n, const int64_t incx, const int64_t incy);

                // Complex Euclidean norm: result = sqrt(sum(|x[i]|^2)) for i = 0 to n-1
// x: complex vector of length n (stored as 2*n floats)
int aclblasScnrm2(std::complex<float> *x, float *result, const int64_t n, const int64_t incx, void *stream);

// ========== CAL系列接口（向量缩放） ==========

// 实数向量缩放: x = alpha * x
// x: 实数向量，长度为n
// alpha: 实数标量
int aclblasSscal(aclblasHandle handle, float *x, const float alpha, const int64_t n, const int64_t incx);

// 复数向量缩放（实数标量）: x = alpha * x
// x: 复数向量，长度为n
// alpha: 实数标量
int aclblasCsscal(aclblasHandle handle, std::complex<float> *x, const float alpha, const int64_t n, const int64_t incx);

// 复数向量缩放（复数标量）: x = alpha * x
// x: 复数向量，长度为n
// alpha: 复数标量
int aclblasCscal(aclblasHandle handle, std::complex<float> *x, const std::complex<float> alpha,
                 const int64_t n, const int64_t incx);

// ========== SWAP系列接口（向量交换） ==========

// 实数向量交换: x <-> y
// x, y: 实数向量，长度为n
int aclblasSswap(aclblasHandle handle, float *x, float *y, const int64_t n, const int64_t incx, const int64_t incy);

// 复数向量交换: x <-> y
// x, y: 复数向量（存储为float数组），长度为n
int aclblasCswap(aclblasHandle handle, float *x, float *y, const int64_t n, const int64_t incx, const int64_t incy);

// ========== AXPY系列接口（向量缩放加法） ==========

// 复向量缩放加法: y = alpha * x + y
// x: 输入复向量，长度为n
// y: 输入/输出复向量，长度为n
// alpha: 复数标量
int aclblasCaxpy(aclblasHandle handle, const std::complex<float> *x, std::complex<float> *y,
                 const std::complex<float> alpha, const int64_t n, const int64_t incx, const int64_t incy);

// ========== GEMV系列接口（矩阵向量乘法） ==========

// 复矩阵向量乘法: y = alpha * A * x + beta * y (或 A^T, A^H)
// A: m x n 复矩阵
// x: 复向量（长度为n当trans=N，长度为m当trans=T/C）
// y: 复向量（长度为m当trans=N，长度为n当trans=T/C）
// trans: 0=N, 1=T(转置), 2=C(共轭转置)
int aclblasCgemv(aclblasHandle handle,
                  aclblasOperation trans,
                  const int64_t m, const int64_t n,
                  const std::complex<float> &alpha,
                  const std::complex<float> *A, const int64_t lda,
                  const std::complex<float> *x, const int64_t incx,
                  const std::complex<float> &beta,
std::complex<float> *y, const int64_t incy,
                   void *stream);

// ========== TRMV系列接口（三角矩阵向量乘法） ==========

// 实三角矩阵向量乘法: x = A * x 或 x = A^T * x
// A: n阶三角矩阵（上三角或下三角）
// x: n维向量
// uplo: 上三角(ACLBLAS_FILL_MODE_UPPER)或下三角(ACLBLAS_FILL_MODE_LOWER)
// trans: 不转置(ACLBLAS_OP_N)或转置(ACLBLAS_OP_T)
// diag: 非单位对角线(ACLBLAS_DIAG_NON_UNIT)或单位对角线(ACLBLAS_DIAG_UNIT)
int aclblasStrmv(aclblasHandle handle,
                 aclblasFillMode uplo,
                 aclblasOperation trans,
                 aclblasDiagType diag,
                 const int64_t n,
                 const float *A, const int64_t lda,
                 float *x, const int64_t incx,
                 void *stream);

// ========== SYR系列接口（对称矩阵秩更新） ==========

// 实对称秩1更新: A = alpha * x * x^T + A
// A: n阶对称矩阵（上三角或下三角）
// x: n维向量
// uplo: 上三角(ACLBLAS_UPPER)或下三角(ACLBLAS_LOWER)
int aclblasSsyr(aclblasHandle handle,
                aclblasFillMode uplo,
                const int64_t n,
                const float alpha,
                const float *x, const int64_t incx,
                float *A, const int64_t lda,
                void *stream);

// 实对称秩2更新: A = alpha * x * y^T + alpha * y * x^T + A
// A: n阶对称矩阵（上三角或下三角）
// x, y: n维向量
// uplo: 上三角(ACLBLAS_UPPER)或下三角(ACLBLAS_LOWER)
int aclblasSsyr2(aclblasHandle handle,
                 aclblasFillMode uplo,
                 const int64_t n,
                 const float alpha,
                 const float *x, const int64_t incx,
                 const float *y, const int64_t incy,
                 float *A, const int64_t lda,
                 void *stream);

// ========== GEMM系列接口（矩阵乘法） ==========

// 批量复矩阵乘法: C[i] = A[i] * B[i] (batchCount个矩阵)
// A: batchCount个m×k复矩阵
// B: batchCount个k×n复矩阵
// C: batchCount个m×n复矩阵
int aclblasCgemmBatched(aclblasHandle handle,
                         const int64_t m, const int64_t k, const int64_t n,
                         const int64_t batchCount,
                         const float *A, const int64_t lda,
                         const float *B, const int64_t ldb,
                         float *C, const int64_t ldc,
                         void *stream);

// 复矩阵乘法: C = alpha * op(A) * op(B) + beta * C
// A: m×k 复矩阵（或k×m当transA=T/C）
// B: k×n 复矩阵（或n×k当transB=T/C）
// C: m×n 复矩阵
// transA: 0=N, 1=T(转置), 2=C(共轭转置)
// transB: 0=N, 1=T(转置), 2=C(共轭转置)
// alpha, beta: 复数标量
int aclblasCgemm(aclblasHandle handle,
                 aclblasOperation transA, aclblasOperation transB,
                 const int64_t m, const int64_t n, const int64_t k,
                 const std::complex<float> *alpha,
                 const float *A, const int64_t lda,
                 const float *B, const int64_t ldb,
                 const std::complex<float> *beta,
                 float *C, const int64_t ldc,
                 void *stream);

// ========== TRMM系列接口（三角矩阵乘法） ==========

// 实三角矩阵乘法: C = alpha * op(A) * op(B)
// A: m×k 实三角矩阵（或k×m当transa=T）
// B: k×n 实矩阵（或n×k当transb=T）
// C: m×n 实矩阵
// side: LEFT(A在左侧)或RIGHT(A在右侧)
// uplo: 上三角(ACLBLAS_UPPER)或下三角(ACLBLAS_LOWER)
// transa: 不转置(ACLBLAS_OP_N)或转置(ACLBLAS_OP_T)
// transb: 不转置(ACLBLAS_OP_N)或转置(ACLBLAS_OP_T)
// diag: 非单位对角线(ACLBLAS_NON_UNIT)或单位对角线(ACLBLAS_UNIT)
int aclblasStrmm(aclblasHandle handle,
                 aclblasSideMode side,
                 aclblasFillMode uplo,
                 aclblasOperation transa,
                 aclblasOperation transb,
                 aclblasDiagType diag,
                 const int64_t m, const int64_t n, const int64_t k,
                 const float alpha,
                 const float *A, const int64_t lda,
                 const float *B, const int64_t ldb,
                 float *C, const int64_t ldc,
                 void *stream);

int aclblasCtrmv(aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
                 aclblasDiagType_t diag, int64_t n,
                 const float *A, int64_t lda, float *x, int64_t incx);