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

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief aclblas status codes definition */
typedef enum aclblasStatus
{
    ACLBLAS_STATUS_SUCCESS = 0,              /**< Function succeeds */
    ACLBLAS_STATUS_NOT_INITIALIZED = 1,      /**< aclblas library not initialized */
    ACLBLAS_STATUS_ALLOC_FAILED = 2,         /**< resource allocation failed */
    ACLBLAS_STATUS_INVALID_VALUE = 3,        /**< unsupported numerical value was passed to function */
    ACLBLAS_STATUS_MAPPING_ERROR = 4,        /**< access to memory space failed */
    ACLBLAS_STATUS_EXECUTION_FAILED = 5,     /**< program failed to execute */
    ACLBLAS_STATUS_INTERNAL_ERROR = 6,       /**< an internal aclblas operation failed */
    ACLBLAS_STATUS_NOT_SUPPORTED = 7,        /**< function not implemented */
    ACLBLAS_STATUS_ARCH_MISMATCH = 8,        /**< architecture mismatch */
    ACLBLAS_STATUS_HANDLE_IS_NULLPTR = 9,    /**< aclblas handle is null pointer */
    ACLBLAS_STATUS_INVALID_ENUM = 10,        /**< unsupported enum value was passed to function */
    ACLBLAS_STATUS_UNKNOWN = 11              /**< back-end returned an unsupported status code */
} aclblasStatus_t;
typedef aclblasStatus_t aclblasLtStatus;

#ifndef ACLBLAS_OPERATION_DECLARED
#define ACLBLAS_OPERATION_DECLARED
/*! \brief Used to specify whether the matrix is to be transposed or not (Fortran BLAS style). */
typedef enum aclblasOperation
{
    ACLBLAS_OP_N = 111, /**< Operate with the matrix */
    ACLBLAS_OP_T = 112, /**< Operate with the transpose of the matrix */
    ACLBLAS_OP_C = 113  /**< Operate with the conjugate transpose of the matrix */
} aclblasOperation_t;

#elif __cplusplus >= 201103L
static_assert(ACLBLAS_OP_N == 111, "Inconsistent declaration of ACLBLAS_OP_N");
static_assert(ACLBLAS_OP_T == 112, "Inconsistent declaration of ACLBLAS_OP_T");
static_assert(ACLBLAS_OP_C == 113, "Inconsistent declaration of ACLBLAS_OP_C");
#endif // ACLBLAS_OPERATION_DECLARED

/*! \brief The compute type to be used. Currently only used with GemmEx.
 *  Note that support for compute types is largely dependent on backend. */
typedef enum aclblasComputeType
{
    ACLBLAS_COMPUTE_16F = 0,           /**< compute will be at least 16-bit precision */
    ACLBLAS_COMPUTE_16F_PEDANTIC = 1,  /**< compute will be exactly 16-bit precision */
    ACLBLAS_COMPUTE_32F = 2,           /**< compute will be at least 32-bit precision */
    ACLBLAS_COMPUTE_32F_PEDANTIC = 3,  /**< compute will be exactly 32-bit precision */
    ACLBLAS_COMPUTE_32F_FAST_16F = 4,  /**< 32-bit input can use 16-bit compute */
    ACLBLAS_COMPUTE_32F_FAST_16BF = 5, /**< 32-bit input is bf16 compute */
    ACLBLAS_COMPUTE_32F_FAST_TF32 = 6, /**< 32-bit input can use tensor cores w/ TF32 compute */
    ACLBLAS_COMPUTE_64F = 7,           /**< compute will be at least 64-bit precision */
    ACLBLAS_COMPUTE_64F_PEDANTIC = 8,  /**< compute will be exactly 64-bit precision */
    ACLBLAS_COMPUTE_32I = 9,           /**< compute will be at least 32-bit integer precision */
    ACLBLAS_COMPUTE_32I_PEDANTIC = 10, /**< compute will be exactly 32-bit integer precision */
} aclblasComputeType_t;

#ifdef __cplusplus
}
#endif
