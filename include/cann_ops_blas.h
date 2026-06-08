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

using aclblasHandle = void*;
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
aclblasStatus_t aclblasCreate(aclblasHandle_t* handle);

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
aclblasStatus_t aclblasGetStream(aclblasHandle_t handle, aclrtStream* stream);

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
aclblasStatus_t aclblasSetWorkspace(aclblasHandle_t handle, void* workspace, size_t workspaceSize);

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
aclblasStatus_t aclblasGetVersion(aclblasHandle_t handle, int* version);

/**
 * @brief 配置日志接口
 * @param logFile  日志文件名称，为空不输出
 * @param logToStdOut 是否输出到标准流
 * @param logToKdlls  是否输出到内核
 * @param logLevel 日志级别
 * @return ACLBLAS_STATUS_SUCCESS 成功
 *         ACLBLAS_STATUS_INVALID_VALUE 参数无效
 */
aclblasStatus_t aclblasLoggerConfigure(
    const char* logFile, bool logToStdOut, bool logToKdlls, aclblasLogLevel_t logLevel);

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

aclblasStatus_t aclblasScopy(
    aclblasHandle_t handle, uint8_t* x, uint8_t* y, const int64_t n, const int64_t incx, const int64_t incy);

aclblasStatus_t aclblasSpmv(
    aclblasHandle_t handle, const float* aPacked, const float* x, const float* y, float* z, const float alpha,
    const float beta, const int64_t n, const int64_t incx, const int64_t incy);

aclblasStatus_t aclblasSspmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x,
    int incx, const float* beta, float* y, int incy);

aclblasStatus_t aclblasSger(
    aclblasHandle_t handle, int64_t m, int64_t n, const float* alpha, const float* x, int64_t incx, float* y,
    int64_t incy, float* A, int64_t lda);

aclblasStatus_t aclblasStrsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* A, int lda, float* x, int incx);

aclblasStatus_t aclblasSsymv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* A, int lda, const float* x,
    int incx, const float* beta, float* y, int incy);

aclblasStatus_t aclblasSsbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta, float* y, int incy);

aclblasStatus_t aclblasStpsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* AP, float* x, int incx);

aclblasStatus_t aclblasStpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* AP, float* x, int incx);

// Early arch22 implementation. Prefer aclblasStpmv for new code; this interface may evolve or be removed in future
// releases.
aclblasStatus_t aclblasStpmv_legacy(
    aclblasHandle_t handle, aclblasFillMode uplo, aclblasOperation trans, aclblasDiagType diag, int64_t n,
    const float* aPacked, const float* x, float* y, int64_t incx);

// Early implementation with a non-standard parameter layout. Prefer aclblasStbmv for new code; this interface may
// evolve or be removed in future releases.
aclblasStatus_t aclblasStbmv_legacy(
    aclblasHandle_t handle, const float* a, const int64_t lda, const float* x, float* y, const int64_t n,
    const int64_t k, const int64_t incx);

aclblasStatus_t aclblasStbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k,
    const float* A, int lda, float* x, int incx);

aclblasStatus_t aclblasCdotu(
    aclblasHandle_t handle, const int64_t n, uint8_t* x, const int64_t incx, uint8_t* y, const int64_t incy,
    uint8_t* result);

aclblasStatus_t aclblasCdotc(
    aclblasHandle_t handle, const int64_t n, uint8_t* x, const int64_t incx, uint8_t* y, const int64_t incy,
    uint8_t* result);

aclblasStatus_t aclblasSasum(aclblasHandle_t handle, int n, const float* x, int incx, float* result);

aclblasStatus_t aclblasIamax(aclblasHandle_t handle, const int64_t n, uint8_t* x, const int64_t incx, uint8_t* result);

aclblasStatus_t aclblasCsrot(
    aclblasHandle_t handle, const int64_t n, uint8_t* x, const int64_t incx, uint8_t* y, const int64_t incy,
    const float c, const float s);

aclblasStatus_t aclblasColwiseMul(
    aclblasHandle_t handle, const int64_t m, const int64_t n, uint8_t* mat, uint8_t* vec, uint8_t* result);

aclblasStatus_t aclblasComplexMatDot(
    aclblasHandle_t handle, const int64_t m, const int64_t n, uint8_t* matx, uint8_t* maty, uint8_t* result);

aclblasStatus_t aclblasCgemvBatched(
    aclblasHandle_t handle, aclblasOperation trans, const int64_t m, const int64_t n, const std::complex<float>& alpha,
    uint8_t* A, const int64_t lda, uint8_t* x, const int64_t incx, const std::complex<float>& beta, uint8_t* y,
    const int64_t incy, const int64_t batchCount);

aclblasStatus_t aclblasCgerc(
    aclblasHandle_t handle, const int64_t m, const int64_t n, const std::complex<float>& alpha, uint8_t* x,
    const int64_t incx, uint8_t* y, const int64_t incy, uint8_t* A, const int64_t lda);

aclblasStatus_t aclblasSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta,
    float* y, int incy, int batchCount);

aclblasStatus_t aclblasHSHgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda,
    const uint16_t* x, int incx, const float* beta,
    uint16_t* y, int incy, int batchCount);

aclblasStatus_t aclblasHSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda,
    const uint16_t* x, int incx, const float* beta,
    float* y, int incy, int batchCount);

aclblasStatus_t aclblasCcopy(
    aclblasHandle_t handle, uint8_t* x, uint8_t* y, const int64_t n, const int64_t incx, const int64_t incy);

aclblasStatus_t aclblasSdot(
    aclblasHandle_t handle, int64_t n, const float* x, int64_t incx, const float* y, int64_t incy, float* result);

aclblasStatus_t aclblasSnrm2(aclblasHandle_t handle, const int64_t n, uint8_t* x, const int64_t incx, uint8_t* result);

aclblasStatus_t aclblasScnrm2(aclblasHandle_t handle, const int64_t n, uint8_t* x, const int64_t incx, uint8_t* result);

aclblasStatus_t aclblasSscal(
    aclblasHandle_t handle, const int64_t n, const float alpha, uint8_t* x, const int64_t incx);

aclblasStatus_t aclblasCsscal(
    aclblasHandle_t handle, const int64_t n, const float alpha, uint8_t* x, const int64_t incx);

aclblasStatus_t aclblasCscal(
    aclblasHandle_t handle, const int64_t n, const std::complex<float> alpha, uint8_t* x, const int64_t incx);

aclblasStatus_t aclblasSswap(aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy);

aclblasStatus_t aclblasCswap(
    aclblasHandle_t handle, const int64_t n, uint8_t* x, const int64_t incx, uint8_t* y, const int64_t incy);

aclblasStatus_t aclblasCaxpy(
    aclblasHandle_t handle, const int64_t n, const std::complex<float> alpha, uint8_t* x, int64_t incx, uint8_t* y,
    int64_t incy);

aclblasStatus_t aclblasCgemv(
    aclblasHandle_t handle, aclblasOperation trans, const int64_t m, const int64_t n, const std::complex<float>& alpha,
    uint8_t* A, const int64_t lda, uint8_t* x, const int64_t incx, const std::complex<float>& beta, uint8_t* y,
    const int64_t incy);

aclblasStatus_t aclblasStrmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* A, int lda, float* x, int incx);

aclblasStatus_t aclblasSsyr(
    aclblasHandle_t handle, aclblasFillMode uplo, const int n, const float* alpha, const float* x, const int incx,
    float* A, const int lda);

aclblasStatus_t aclblasSsyr2(
    aclblasHandle handle, aclblasFillMode uplo, const int n, const float* alpha, const float* x, const int incx,
    const float* y, const int incy, float* A, const int lda);

aclblasStatus_t aclblasCtrmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int64_t n,
    uint8_t* A, int64_t lda, uint8_t* x, int64_t incx);

aclblasStatus_t aclblasSgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* A,
    int lda, const float* x, int incx, const float* beta, float* y, int incy);

aclblasStatus_t aclblasStpttr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* AP, float* A, int lda);

aclblasStatus_t aclblasStrttp(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* A, int lda, float* AP);

aclblasStatus_t aclblasSrotm(
    aclblasHandle handle, float* x, float* y, const float* sparam, const int64_t n, const int64_t incx,
    const int64_t incy);

aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta, float* y, int incy);


aclblasStatus_t aclblasSgeqrfBatched(
    aclblasHandle_t handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int* info,
    int batchSize);

#ifdef __cplusplus
}
#endif
