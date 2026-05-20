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

aclblasStatus_t aclblasScopy(aclblasHandle handle, uint8_t *x, uint8_t *y, const int64_t n, const int64_t incx, const int64_t incy);

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

// Complex unconjugated dot product: result = Σ(x[i] * y[i]) (no conjugate)
// x, y: complex vectors of length n (stored as 2*n floats)
aclblasStatus_t aclblasCdotu(aclblasHandle handle, const int64_t n, uint8_t *x, const int64_t incx, uint8_t *y, const int64_t incy, uint8_t *result);

aclblasStatus_t aclblasCdotc(aclblasHandle handle, const int64_t n, uint8_t *x, const int64_t incx, uint8_t *y, const int64_t incy, uint8_t *result);

aclblasStatus_t aclblasSasum(aclblasHandle handle, const int64_t n, uint8_t *x, const int64_t incx, uint8_t *result);

aclblasStatus_t aclblasIamax(aclblasHandle handle, const int64_t n, uint8_t *x, const int64_t incx, uint8_t *result);

aclblasStatus_t aclblasCsrot(aclblasHandle handle, const int64_t n, uint8_t *x, const int64_t incx, uint8_t *y, const int64_t incy, const float c, const float s);

aclblasStatus_t aclblasColwiseMul(aclblasHandle handle, const int64_t m, const int64_t n, uint8_t *mat, uint8_t *vec, uint8_t *result);

aclblasStatus_t aclblasComplexMatDot(aclblasHandle handle, const int64_t m, const int64_t n, uint8_t *matx, uint8_t *maty, uint8_t *result);

aclblasStatus_t aclblasCgemvBatched(aclblasHandle handle,
                                     aclblasOperation trans,
                                     const int64_t m, const int64_t n,
                                     const std::complex<float> &alpha,
                                     uint8_t *A, const int64_t lda,
                                     uint8_t *x, const int64_t incx,
                                     const std::complex<float> &beta,
                                     uint8_t *y, const int64_t incy,
                                     const int64_t batchCount);

aclblasStatus_t aclblasCgerc(aclblasHandle handle, const int64_t m, const int64_t n, const std::complex<float> &alpha,
                              uint8_t *x, const int64_t incx, uint8_t *y, const int64_t incy, uint8_t *A, const int64_t lda);

aclblasStatus_t aclblasCcopy(aclblasHandle handle, uint8_t *x, uint8_t *y, const int64_t n, const int64_t incx, const int64_t incy);


aclblasStatus_t aclblasSdot(aclblasHandle handle, const int64_t n, uint8_t *x, const int64_t incx, 
                              uint8_t *y, const int64_t incy, uint8_t *result);

                aclblasStatus_t aclblasSnrm2(aclblasHandle handle, const int64_t n, uint8_t *x, const int64_t incx, uint8_t *result);

aclblasStatus_t aclblasScnrm2(aclblasHandle handle, const int64_t n, uint8_t *x, const int64_t incx, uint8_t *result);

aclblasStatus_t aclblasSscal(aclblasHandle handle, const int64_t n, const float alpha, uint8_t *x, const int64_t incx);

aclblasStatus_t aclblasCsscal(aclblasHandle handle, const int64_t n, const float alpha, uint8_t *x, const int64_t incx);

aclblasStatus_t aclblasCscal(aclblasHandle handle, const int64_t n, const std::complex<float> alpha, uint8_t *x, const int64_t incx);

aclblasStatus_t aclblasSswap(aclblasHandle handle, const int64_t n, uint8_t *x, const int64_t incx, uint8_t *y, const int64_t incy);

aclblasStatus_t aclblasCswap(aclblasHandle handle, const int64_t n, uint8_t *x, const int64_t incx, uint8_t *y, const int64_t incy);

aclblasStatus_t aclblasCaxpy(aclblasHandle handle, const int64_t n, const std::complex<float> alpha, uint8_t *x, int64_t incx, uint8_t *y, int64_t incy);

aclblasStatus_t aclblasCgemv(aclblasHandle handle,
                              aclblasOperation trans,
                              const int64_t m, const int64_t n,
                              const std::complex<float> &alpha,
                              uint8_t *A, const int64_t lda,
                              uint8_t *x, const int64_t incx,
                              const std::complex<float> &beta,
                              uint8_t *y, const int64_t incy);

aclblasStatus_t aclblasStrmv(aclblasHandle handle,
                             aclblasFillMode uplo,
                             aclblasOperation trans,
                             aclblasDiagType diag,
                             const int64_t n,
                             uint8_t *A, const int64_t lda,
                             uint8_t *x, const int64_t incx);

aclblasStatus_t aclblasSsyr(aclblasHandle handle,
                            aclblasFillMode uplo,
                            const int64_t n,
                            const float alpha,
                            uint8_t *x, const int64_t incx,
                            uint8_t *A, const int64_t lda);

aclblasStatus_t aclblasSsyr2(aclblasHandle handle,
                             aclblasFillMode uplo,
                             const int64_t n,
                             const float alpha,
                             uint8_t *x, const int64_t incx,
                             uint8_t *y, const int64_t incy,
                             uint8_t *A, const int64_t lda);

aclblasStatus_t aclblasCgemmBatched(aclblasHandle handle,
                                     aclblasOperation transa, aclblasOperation transb,
                                     const int64_t m, const int64_t n, const int64_t k,
                                     const std::complex<float> &alpha,
                                     uint8_t *A, const int64_t lda,
                                     uint8_t *B, const int64_t ldb,
                                     const std::complex<float> &beta,
                                     uint8_t *C, const int64_t ldc,
                                     const int64_t batchCount);

// 复矩阵乘法: C = alpha * op(A) * op(B) + beta * C
// A: m×k 复矩阵（或k×m当transA=T/C）
// B: k×n 复矩阵（或n×k当transB=T/C）
// C: m×n 复矩阵
// transA: 0=N, 1=T(转置), 2=C(共轭转置)
// transB: 0=N, 1=T(转置), 2=C(共轭转置)
// alpha, beta: 复数标量
aclblasStatus_t aclblasCgemm(aclblasHandle handle,
                              aclblasOperation transA, aclblasOperation transB,
                              const int64_t m, const int64_t n, const int64_t k,
                              const std::complex<float> &alpha,
                              uint8_t *A, const int64_t lda,
                              uint8_t *B, const int64_t ldb,
                              const std::complex<float> &beta,
                              uint8_t *C, const int64_t ldc);

aclblasStatus_t aclblasStrmm(aclblasHandle handle,
                             aclblasSideMode side,
                             aclblasFillMode uplo,
                             aclblasOperation trans,
                             aclblasDiagType diag,
                             const int64_t m, const int64_t n, const float alpha,
                             uint8_t *A, const int64_t lda,
                             uint8_t *B, const int64_t ldb,
                             uint8_t *C, const int64_t ldc);

aclblasStatus_t aclblasCtrmv(aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
                             aclblasDiagType_t diag, int64_t n,
                             uint8_t *A, int64_t lda, uint8_t *x, int64_t incx);

/**
 * @brief 带状矩阵向量乘（SGBMV）
 *
 * 计算 y = alpha * op(A) * x + beta * y，其中 A 为 M x N 带状矩阵
 *
 * @param handle aclblas 句柄
 * @param trans  op(A) 操作类型（N=不转置，T=转置，C=共轭转置）
 * @param m      矩阵 A 的行数
 * @param n      矩阵 A 的列数
 * @param kl     次对角线数
 * @param ku     超对角线数
 * @param alpha  标量系数
 * @param A      带状矩阵，紧凑存储格式 (KL+KU+1) x N（列主序）
 * @param lda    A 的前导维度（>= KL+KU+1）
 * @param x      输入向量
 * @param incx   x 元素步长
 * @param beta   标量系数
 * @param y      输入/输出向量
 * @param incy   y 元素步长
 * @return ACLBLAS_STATUS_SUCCESS 成功，或错误状态码
 */
aclblasStatus_t aclblasSgbmv(aclblasHandle handle,
                             aclblasOperation_t trans,
                             int m,
                             int n,
                             int kl,
                             int ku,
                             const float *alpha,
                             const float *A,
                             int lda,
                             const float *x,
                             int incx,
                             const float *beta,
                             float *y,
                             int incy);