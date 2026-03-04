/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cann_ops_blasLt.h"

#include <acl/acl.h>
#include <cstdlib>
#include <cstring>
#include <new>


#define GM_ADDR uint8_t*

namespace {

constexpr int kAclblasLtVersionMajor = 1;
constexpr int kAclblasLtVersionMinor = 0;
constexpr int kAclblasLtVersionPatch = 0;

struct LtHandle {
  uint32_t magic = 0x4C54484C; // 'LTHL'
  int32_t deviceId = 0;
  bool initialized = false;
};

struct MatrixLayout {
  aclDataType type;
  uint64_t rows;
  uint64_t cols;
  int64_t ld;
  aclblasLtOrder_t order;
  int32_t batchCount = 1;
  int64_t stridedBatchOffset = 0;
};

struct MatmulDesc {
  aclblasComputeType_t computeType;
  aclDataType scaleType;
  aclblasLtEpilogue_t epilogue;
  const void* bias;
  aclblasOperation_t transA = ACLBLAS_OP_N;
  aclblasOperation_t transB = ACLBLAS_OP_N;
  aclDataType biasDataType = ACL_FLOAT;
};

struct Preference {
  size_t maxWorkspaceBytes = 0;
  uint32_t searchMode = 0;
};

template <typename T>
static aclblasStatus_t AllocHandle(T** out)
{
  if (out == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }
  *out = nullptr;
  T* p = new (std::nothrow) T();
  if (p == nullptr) {
    return ACLBLAS_STATUS_ALLOC_FAILED;
  }
  *out = p;
  return ACLBLAS_STATUS_SUCCESS;
}

template <typename T>
static aclblasStatus_t FreeHandle(T* p)
{
  if (p == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }
  delete p;
  return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

extern void matmul_kernel_do(GM_ADDR a,
  GM_ADDR b,
  GM_ADDR c,
  GM_ADDR d,
  uint32_t m,
  uint32_t k,
  uint32_t n,
  uint32_t numBlocks,
  void *stream);

extern "C" {

aclblasStatus_t aclblasLtGetVersion(size_t* version)
{
  if (version == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  *version = (static_cast<size_t>(kAclblasLtVersionMajor) << 24) |
             (static_cast<size_t>(kAclblasLtVersionMinor) << 16) |
             static_cast<size_t>(kAclblasLtVersionPatch);
  return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtGetProperty(aclblasLtPropertyType_t type, int* value)
{
  if (value == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  switch (type) {
    case ACLBLASLT_PROPERTY_MAJOR_VERSION:
      *value = kAclblasLtVersionMajor;
      return ACLBLAS_STATUS_SUCCESS;
    case ACLBLASLT_PROPERTY_MINOR_VERSION:
      *value = kAclblasLtVersionMinor;
      return ACLBLAS_STATUS_SUCCESS;
    case ACLBLASLT_PROPERTY_PATCH_LEVEL:
      *value = kAclblasLtVersionPatch;
      return ACLBLAS_STATUS_SUCCESS;
    default:
      return ACLBLAS_STATUS_INVALID_VALUE;
  }
}

aclblasStatus_t aclblasLtCreate(aclblasLtHandle_t* handle)
{
  if (handle == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  LtHandle* h = nullptr;
  auto st = AllocHandle(&h);
  if (st != ACLBLAS_STATUS_SUCCESS) {
    return st;
  }

  // Get current device ID
  int32_t deviceId = 0;
  aclError aclRet = aclrtGetDevice(&deviceId);
  if (aclRet != ACL_SUCCESS) {
    delete h;
    return ACLBLAS_STATUS_NOT_INITIALIZED;
  }

  h->deviceId = deviceId;
  h->initialized = true;
  *handle = reinterpret_cast<aclblasLtHandle_t>(h);
  return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtDestroy(const aclblasLtHandle_t handle)
{
  if (handle == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  auto* h = reinterpret_cast<LtHandle*>(handle);
  if (h->magic != 0x4C54484C) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  h->initialized = false;
  return FreeHandle(h);
}

aclblasStatus_t aclblasLtMatrixLayoutCreate(aclblasLtMatrixLayout_t* layout,
                                            aclDataType type,
                                            uint64_t rows,
                                            uint64_t cols,
                                            int64_t ld)
{
  if (layout == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  if (rows == 0 || cols == 0) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  // ld must be >= rows for column major, >= cols for row major
  // For now, allow ld == 0 to use default
  if (ld < 0) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  MatrixLayout* l = nullptr;
  auto st = AllocHandle(&l);
  if (st != ACLBLAS_STATUS_SUCCESS) {
    return st;
  }

  l->type = type;
  l->rows = rows;
  l->cols = cols;
  l->ld = (ld == 0) ? static_cast<int64_t>(rows) : ld; // Default ld = rows for col-major
  l->order = ACLBLASLT_ORDER_COL;
  l->batchCount = 1;
  l->stridedBatchOffset = 0;

  *layout = reinterpret_cast<aclblasLtMatrixLayout_t>(l);
  return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatrixLayoutDestroy(const aclblasLtMatrixLayout_t layout)
{
  if (layout == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }
  auto* l = reinterpret_cast<MatrixLayout*>(layout);
  return FreeHandle(l);
}

aclblasStatus_t aclblasLtMatrixLayoutSetAttribute(aclblasLtMatrixLayout_t layout,
                                                   aclblasLtMatrixLayoutAttribute_t attr,
                                                   const void* buf,
                                                   size_t sizeInBytes)
{
  if (layout == nullptr || buf == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  auto* l = reinterpret_cast<MatrixLayout*>(layout);

  switch (attr) {
    case ACLBLASLT_MATRIX_LAYOUT_ORDER: {
      if (sizeInBytes != sizeof(aclblasLtOrder_t) && sizeInBytes != sizeof(int32_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      int32_t v = 0;
      std::memcpy(&v, buf, sizeof(int32_t));
      l->order = static_cast<aclblasLtOrder_t>(v);
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT: {
      if (sizeInBytes != sizeof(int32_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      std::memcpy(&l->batchCount, buf, sizeof(int32_t));
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET: {
      if (sizeInBytes != sizeof(int64_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      std::memcpy(&l->stridedBatchOffset, buf, sizeof(int64_t));
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATRIX_LAYOUT_ROWS: {
      if (sizeInBytes != sizeof(uint64_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      std::memcpy(&l->rows, buf, sizeof(uint64_t));
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATRIX_LAYOUT_COLS: {
      if (sizeInBytes != sizeof(uint64_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      std::memcpy(&l->cols, buf, sizeof(uint64_t));
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATRIX_LAYOUT_LD: {
      if (sizeInBytes != sizeof(int64_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      std::memcpy(&l->ld, buf, sizeof(int64_t));
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATRIX_LAYOUT_TYPE: {
      if (sizeInBytes != sizeof(int32_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      int32_t typeVal = 0;
      std::memcpy(&typeVal, buf, sizeof(int32_t));
      l->type = static_cast<aclDataType>(typeVal);
      return ACLBLAS_STATUS_SUCCESS;
    }
    default:
      return ACLBLAS_STATUS_NOT_SUPPORTED;
  }
}

aclblasStatus_t aclblasLtMatrixLayoutGetAttribute(const aclblasLtMatrixLayout_t layout,
                                                   aclblasLtMatrixLayoutAttribute_t attr,
                                                   void* buf,
                                                   size_t sizeInBytes,
                                                   size_t* sizeWritten)
{
  if (layout == nullptr || buf == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  auto* l = reinterpret_cast<const MatrixLayout*>(layout);

  switch (attr) {
    case ACLBLASLT_MATRIX_LAYOUT_ORDER: {
      if (sizeInBytes < sizeof(aclblasLtOrder_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      std::memcpy(buf, &l->order, sizeof(aclblasLtOrder_t));
      if (sizeWritten != nullptr) {
        *sizeWritten = sizeof(aclblasLtOrder_t);
      }
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT: {
      if (sizeInBytes < sizeof(int32_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      std::memcpy(buf, &l->batchCount, sizeof(int32_t));
      if (sizeWritten != nullptr) {
        *sizeWritten = sizeof(int32_t);
      }
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET: {
      if (sizeInBytes < sizeof(int64_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      std::memcpy(buf, &l->stridedBatchOffset, sizeof(int64_t));
      if (sizeWritten != nullptr) {
        *sizeWritten = sizeof(int64_t);
      }
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATRIX_LAYOUT_ROWS: {
      if (sizeInBytes < sizeof(uint64_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      std::memcpy(buf, &l->rows, sizeof(uint64_t));
      if (sizeWritten != nullptr) {
        *sizeWritten = sizeof(uint64_t);
      }
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATRIX_LAYOUT_COLS: {
      if (sizeInBytes < sizeof(uint64_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      std::memcpy(buf, &l->cols, sizeof(uint64_t));
      if (sizeWritten != nullptr) {
        *sizeWritten = sizeof(uint64_t);
      }
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATRIX_LAYOUT_LD: {
      if (sizeInBytes < sizeof(int64_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      std::memcpy(buf, &l->ld, sizeof(int64_t));
      if (sizeWritten != nullptr) {
        *sizeWritten = sizeof(int64_t);
      }
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATRIX_LAYOUT_TYPE: {
      if (sizeInBytes < sizeof(aclDataType)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      std::memcpy(buf, &l->type, sizeof(aclDataType));
      if (sizeWritten != nullptr) {
        *sizeWritten = sizeof(aclDataType);
      }
      return ACLBLAS_STATUS_SUCCESS;
    }
    default:
      return ACLBLAS_STATUS_NOT_SUPPORTED;
  }
}

aclblasStatus_t aclblasLtMatmulDescCreate(aclblasLtMatmulDesc_t* desc,
                                          aclblasComputeType_t computeType,
                                          aclDataType scaleType)
{
  if (desc == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  MatmulDesc* d = nullptr;
  auto st = AllocHandle(&d);
  if (st != ACLBLAS_STATUS_SUCCESS) {
    return st;
  }

  d->computeType = computeType;
  d->scaleType = scaleType;
  d->epilogue = ACLBLASLT_EPILOGUE_DEFAULT;
  d->bias = nullptr;
  d->transA = ACLBLAS_OP_N;
  d->transB = ACLBLAS_OP_N;
  d->biasDataType = ACL_FLOAT;

  *desc = reinterpret_cast<aclblasLtMatmulDesc_t>(d);
  return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulDescDestroy(const aclblasLtMatmulDesc_t desc)
{
  if (desc == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }
  auto* d = reinterpret_cast<MatmulDesc*>(desc);
  return FreeHandle(d);
}

aclblasStatus_t aclblasLtMatmulDescSetAttribute(aclblasLtMatmulDesc_t desc,
                                                 aclblasLtMatmulDescAttribute_t attr,
                                                 const void* buf,
                                                 size_t sizeInBytes)
{
  if (desc == nullptr || buf == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  auto* d = reinterpret_cast<MatmulDesc*>(desc);

  switch (attr) {
    case ACLBLASLT_MATMUL_DESC_EPILOGUE: {
      if (sizeInBytes != sizeof(aclblasLtEpilogue_t) && sizeInBytes != sizeof(uint32_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      uint32_t v = 0;
      std::memcpy(&v, buf, sizeof(uint32_t));
      d->epilogue = static_cast<aclblasLtEpilogue_t>(v);
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATMUL_DESC_BIAS_POINTER: {
      if (sizeInBytes != sizeof(void*)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      std::memcpy(&d->bias, buf, sizeof(void*));
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATMUL_DESC_TRANSA: {
      if (sizeInBytes != sizeof(int32_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      int32_t v = 0;
      std::memcpy(&v, buf, sizeof(int32_t));
      d->transA = static_cast<aclblasOperation_t>(v);
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATMUL_DESC_TRANSB: {
      if (sizeInBytes != sizeof(int32_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      int32_t v = 0;
      std::memcpy(&v, buf, sizeof(int32_t));
      d->transB = static_cast<aclblasOperation_t>(v);
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATMUL_DESC_BIAS_DATA_TYPE: {
      if (sizeInBytes != sizeof(int32_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      int32_t v = 0;
      std::memcpy(&v, buf, sizeof(int32_t));
      d->biasDataType = static_cast<aclDataType>(v);
      return ACLBLAS_STATUS_SUCCESS;
    }
    default:
      return ACLBLAS_STATUS_NOT_SUPPORTED;
  }
}

aclblasStatus_t aclblasLtMatmulPreferenceCreate(aclblasLtMatmulPreference_t* pref)
{
  if (pref == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  Preference* p = nullptr;
  auto st = AllocHandle(&p);
  if (st != ACLBLAS_STATUS_SUCCESS) {
    return st;
  }

  p->maxWorkspaceBytes = 0;
  p->searchMode = 0;

  *pref = reinterpret_cast<aclblasLtMatmulPreference_t>(p);
  return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulPreferenceDestroy(const aclblasLtMatmulPreference_t pref)
{
  if (pref == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }
  auto* p = reinterpret_cast<Preference*>(pref);
  return FreeHandle(p);
}

aclblasStatus_t aclblasLtMatmulPreferenceSetAttribute(aclblasLtMatmulPreference_t pref,
                                                      aclblasLtMatmulPreferenceAttribute_t attr,
                                                      const void* buf,
                                                      size_t sizeInBytes)
{
  if (pref == nullptr || buf == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  auto* p = reinterpret_cast<Preference*>(pref);

  switch (attr) {
    case ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: {
      if (sizeInBytes != sizeof(uint64_t) && sizeInBytes != sizeof(size_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      uint64_t v = 0;
      std::memcpy(&v, buf, sizeInBytes);
      p->maxWorkspaceBytes = static_cast<size_t>(v);
      return ACLBLAS_STATUS_SUCCESS;
    }
    case ACLBLASLT_MATMUL_PREF_SEARCH_MODE: {
      if (sizeInBytes != sizeof(uint32_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
      }
      std::memcpy(&p->searchMode, buf, sizeof(uint32_t));
      return ACLBLAS_STATUS_SUCCESS;
    }
    default:
      return ACLBLAS_STATUS_NOT_SUPPORTED;
  }
}

aclblasStatus_t aclblasLtMatmulAlgoGetHeuristic(aclblasLtHandle_t handle,
                                                 aclblasLtMatmulDesc_t matmulDesc,
                                                 aclblasLtMatrixLayout_t Adesc,
                                                 aclblasLtMatrixLayout_t Bdesc,
                                                 aclblasLtMatrixLayout_t Cdesc,
                                                 aclblasLtMatrixLayout_t Ddesc,
                                                 aclblasLtMatmulPreference_t pref,
                                                 int requestedAlgoCount,
                                                 aclblasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                                 int* returnAlgoCount)
{
  // Validate input parameters
  if (returnAlgoCount == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }
  *returnAlgoCount = 0;

  if (requestedAlgoCount <= 0 || heuristicResultsArray == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  if (handle == nullptr || matmulDesc == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  if (Adesc == nullptr || Bdesc == nullptr || Cdesc == nullptr || Ddesc == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  // Get workspace size from preference
  size_t maxWorkspace = 0;
  if (pref != nullptr) {
    auto* p = reinterpret_cast<Preference*>(pref);
    maxWorkspace = p->maxWorkspaceBytes;
  }

  // Get matrix dimensions
  auto* A = reinterpret_cast<MatrixLayout*>(Adesc);
  auto* B = reinterpret_cast<MatrixLayout*>(Bdesc);
  auto* D = reinterpret_cast<MatrixLayout*>(Ddesc);
  auto* desc = reinterpret_cast<MatmulDesc*>(matmulDesc);

  // Validate dimensions for GEMM: D = A * B + C
  // A: m x k, B: k x n, C/D: m x n
  uint64_t m = D->rows;
  uint64_t n = D->cols;
  uint64_t k = (desc->transA == ACLBLAS_OP_N) ? A->cols : A->rows;

  // Basic validation
  if (A->type != B->type) {
    heuristicResultsArray[0].state = ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  // Fill heuristic result
  // heuristicResultsArray[0].algo.algoId = 0;
  heuristicResultsArray[0].algo.max_workspace_bytes = maxWorkspace;
  heuristicResultsArray[0].workspaceSize = maxWorkspace;
  heuristicResultsArray[0].state = ACLBLAS_STATUS_SUCCESS;
  heuristicResultsArray[0].wavesCount = 1.0f;
  memset(heuristicResultsArray[0].reserved, 0, sizeof(heuristicResultsArray[0].reserved));

  *returnAlgoCount = 1;
  return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmul(aclblasLtHandle_t handle,
                                aclblasLtMatmulDesc_t matmulDesc,
                                const void* alpha,
                                const void* A,
                                aclblasLtMatrixLayout_t Adesc,
                                const void* B,
                                aclblasLtMatrixLayout_t Bdesc,
                                const void* beta,
                                const void* C,
                                aclblasLtMatrixLayout_t Cdesc,
                                void* D,
                                aclblasLtMatrixLayout_t Ddesc,
                                const aclblasLtMatmulAlgo_t* algo,
                                void* workspace,
                                size_t workspaceSizeInBytes,
                                aclrtStream stream)
{
  // Validate handle
  if (handle == nullptr) {
    return ACLBLAS_STATUS_NOT_INITIALIZED;
  }

  auto* h = reinterpret_cast<LtHandle*>(handle);
  if (!h->initialized || h->magic != 0x4C54484C) {
    return ACLBLAS_STATUS_NOT_INITIALIZED;
  }

  // Validate descriptors
  if (matmulDesc == nullptr || Adesc == nullptr || Bdesc == nullptr ||
      Cdesc == nullptr || Ddesc == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  // Validate pointers
  if (alpha == nullptr || beta == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  if (A == nullptr || B == nullptr || D == nullptr) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  // Get layout info
  auto* ALayout = reinterpret_cast<MatrixLayout*>(Adesc);
  auto* BLayout = reinterpret_cast<MatrixLayout*>(Bdesc);
  auto* CLayout = reinterpret_cast<MatrixLayout*>(Cdesc);
  auto* DLayout = reinterpret_cast<MatrixLayout*>(Ddesc);
  auto* desc = reinterpret_cast<MatmulDesc*>(matmulDesc);

  // Get dimensions
  uint64_t m = DLayout->rows;
  uint64_t n = DLayout->cols;
  uint64_t k = 0;

  // Determine k based on transpose operations
  if (desc->transA == ACLBLAS_OP_N) {
    k = ALayout->cols;
  } else {
    k = ALayout->rows;
  }

  // Validate workspace alignment (must be 16B aligned)
  if (workspace != nullptr && (reinterpret_cast<uintptr_t>(workspace) & 0xF) != 0) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  // Validate workspace size
  if (algo != nullptr && workspaceSizeInBytes < algo->max_workspace_bytes) {
    return ACLBLAS_STATUS_INVALID_VALUE;
  }

  // Actual GEMM implementation using CANN runtime
  uint32_t numBlocks = 24;
  matmul_kernel_do(static_cast<GM_ADDR>(const_cast<void*>(A)),
                   static_cast<GM_ADDR>(const_cast<void*>(B)),
                   static_cast<GM_ADDR>(const_cast<void*>(C)),
                   static_cast<GM_ADDR>(const_cast<void*>(D)),
                   m,
                   k,
                   n,
                   numBlocks,
                   stream);

  return ACLBLAS_STATUS_SUCCESS;
}

} // extern "C"