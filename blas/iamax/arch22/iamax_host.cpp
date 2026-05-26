/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use the License for the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file iamax_host.cpp
 * \brief Host side implementation for iamax operator
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/kernel_launch/aclblas_kernel_do.h"
#include "common/helper/aclblas_handle_internal.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// Constants from original tiling implementation
constexpr int32_t MAXNUMF32ELEEACHCORE =
    23040; // å®æ°è¶è¿è¿ä¹å¤ä¸ªï¼éè¦å¤è½®å¾ªç¯å¤ç?
constexpr int32_t MAX_NUM_COM_ELE_EACH_CORE =
    11520; // å¤æ°è¶è¿è¿ä¹å¤ä¸ªï¼éè¦å¤è½®å¾ªç¯å¤ç?
constexpr int32_t BYTESPERBLOCK = 32;
constexpr int32_t BYTESPERREPEAT = 256;
constexpr int32_t F32LEN = 4;
constexpr uint32_t MAXVECTORNUM = 40;

constexpr int32_t GM_RESULT_LEN = 2;
constexpr int32_t BYTE_LEN_4 = 4;
constexpr uint64_t ELEMENTS_IN_BLOCK = 8;
constexpr int32_t BLOCKS_PER_REPEAT = 8;
constexpr uint64_t MAX_COER_IN_REPEATE = 32;
constexpr int32_t MAX_REPEATS = 255;

// å¦æè¾å¥æ°æ®å¾å¤§ï¼UBéå¤æ¬¡å¤çï¼å¹¶ä¸è½®æ¬¡å¾å¤ï¼ä¸è½ç­ææè½®æ¬¡é½æ±æ»å®ï¼è¿æ ·ä¸­é´ç»æå¤ªå åå­ï¼éè¦åæ¶å¯¹é¨åä¸­é´ç»æåreduceMax
// ä¸­é´ç»ææ?kè§åï¼å³2*1024/32=64æ¬¡ï¼å³åæ ¸è¶è¿?4æ¬?å°±è¦åä¸æ¬¡reduceMaxï¼å°ç©ºé´å ç¨éä¸ºä¸ä¸ªblock
// 64-63æ¯å ä¸ºæä¸ä¸ªæ¯åå²åç¼©ç»æ
constexpr int32_t DEAL_TIMES_EACH_CORE_REDUCE = 63;

// Tiling data structure matching the kernel expectations
struct IamaxTilingData {
    uint32_t incx;
    uint32_t needVecCoreNum;
    uint32_t dtypeFlag;
    uint32_t rstLenAllCoreBytes;
    uint32_t tailCount;
    uint32_t maxRepeatLen;
    uint32_t startOffset[MAXVECTORNUM];
    uint32_t eleTotalEachCore[MAXVECTORNUM];
    uint32_t dealTimesEachCore[MAXVECTORNUM];
    uint32_t dealLenEachTime[MAXVECTORNUM];
    uint32_t reduceMaxRstsLenEachCore[MAXVECTORNUM];
    uint32_t dealLenUpBlockEachTime[MAXVECTORNUM];
    uint32_t totalRptCntNor[MAXVECTORNUM];
    uint32_t totalRptCntNorRemainder[MAXVECTORNUM];
    uint32_t rptBatchCntNor[MAXVECTORNUM];
    uint32_t rptBatchCntNorRemainder[MAXVECTORNUM];
    uint32_t rmdRptLenNor[MAXVECTORNUM];
};

// Helper function: ceiling division
uint32_t CeilA2B(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

// Determine the number of vector cores needed
uint32_t GetNeedVecCoreNum(uint32_t tensorLen, uint32_t elements)
{
    uint32_t needVecCoreNum = 1;
    if (tensorLen > MAXVECTORNUM * elements) {
        needVecCoreNum = MAXVECTORNUM;
    } else if (tensorLen > elements) {
        needVecCoreNum =
            tensorLen /
            elements; // ä¸è¶³ä¸ä¸ªrepeatçæ°æ®ç»ç¬¬ä¸ä¸ªæ ¸ï¼åå°å¤æ ¸åæ­¥çå¯è½æ?
    } else {
        needVecCoreNum = 1;
    }
    return needVecCoreNum;
}

// Full tiling calculation for iamax - migrated from original implementation
IamaxTilingData CalIamaxTilingData(uint32_t n, uint32_t incx, uint32_t vecCoreNum, uint32_t flag)
{
    IamaxTilingData tilingData;
    memset(&tilingData, 0, sizeof(IamaxTilingData));

    uint32_t numberElements = n;
    uint32_t dtypeFlag = flag;
    uint32_t elementsPerRepeat = BYTESPERREPEAT / F32LEN;

    uint32_t tensorLen = n;

    // Handle complex numbers
    uint32_t complexNum = 2;
    if (dtypeFlag == 1) {
        tensorLen = tensorLen * complexNum;
        numberElements = numberElements * complexNum;
    }

    uint32_t minEleEachCore = elementsPerRepeat;
    uint32_t tmpLen = (tensorLen < numberElements) ? tensorLen : numberElements;
    uint32_t needVecCoreNum = GetNeedVecCoreNum(tmpLen, minEleEachCore);
    if (needVecCoreNum == 0) {
        needVecCoreNum = 1;
    }
    uint32_t rstLenAllCoreBytes = needVecCoreNum * GM_RESULT_LEN * BYTE_LEN_4;

    // ærepeat64ååï¼å°½éä¿è¯æ¯ä¸ªæ ¸åå°æ´repeatçæ°æ®ï¼å°¾åæ°æ®é¨åä¸¢ç»å¤´åæ ?
    uint32_t minEleRepeatsNumber = tmpLen / minEleEachCore;
    uint32_t minEleRepeatTail = tmpLen % minEleEachCore;

    uint32_t minEleRepeatsNumberEachCore = minEleRepeatsNumber / needVecCoreNum;
    uint32_t minEleRepeatsNumbeTail = minEleRepeatsNumber % needVecCoreNum;

    // Allocate temporary arrays for tiling calculation
    uint32_t* allMem = new uint32_t[12 * MAXVECTORNUM];

    uint32_t* startOffset = allMem;
    uint32_t* endOffset = allMem + MAXVECTORNUM;
    uint32_t* eleTotalEachCore = allMem + 2 * MAXVECTORNUM;
    uint32_t* dealLenEachTime = allMem + 3 * MAXVECTORNUM;
    uint32_t* dealTimesEachCore = allMem + 4 * MAXVECTORNUM;
    uint32_t* reduceMaxRstsLenEachCore = allMem + 5 * MAXVECTORNUM;
    uint32_t* dealLenUpBlockEachTime = allMem + 6 * MAXVECTORNUM;
    uint32_t* totalRptCntNor = allMem + 7 * MAXVECTORNUM;
    uint32_t* totalRptCntNorRemainder = allMem + 8 * MAXVECTORNUM;
    uint32_t* rptBatchCntNor = allMem + 9 * MAXVECTORNUM;
    uint32_t* rptBatchCntNorRemainder = allMem + 10 * MAXVECTORNUM;
    uint32_t* rmdRptLenNor = allMem + 11 * MAXVECTORNUM;

    uint32_t eleLenEachCore = 0;
    for (uint32_t i = 0; i < needVecCoreNum; i++) {
        eleLenEachCore = minEleRepeatsNumberEachCore * minEleEachCore;
        if (i == 0) {
            startOffset[i] = 0;
        } else {
            startOffset[i] = endOffset[i - 1];
        }

        // ååç»æææ ¸
        if (minEleRepeatsNumbeTail > 0) {
            eleLenEachCore += minEleEachCore;
            minEleRepeatsNumbeTail--;
        }
        dealTimesEachCore[i] = 0;
        dealLenEachTime[i] = eleLenEachCore; // ä¸å¸¦å°¾åç®?
        if (eleLenEachCore > 0 && eleLenEachCore <= MAXNUMF32ELEEACHCORE) {
            dealTimesEachCore[i] = 1;
        } else if (eleLenEachCore > MAXNUMF32ELEEACHCORE) {
            dealTimesEachCore[i] = CeilA2B(eleLenEachCore, MAXNUMF32ELEEACHCORE);
            dealLenEachTime[i] = MAXNUMF32ELEEACHCORE;
        } else {
            dealTimesEachCore[i] = 0;
            dealLenEachTime[i] = 0;
        }

        uint32_t dealLenEachTimeAttachTail = dealLenEachTime[i];
        if (i == 0 && minEleRepeatTail != 0) {
            eleLenEachCore += minEleRepeatTail; // å°¾åå¨ç»ç¬¬ä¸ä¸ªæ ¸
            if (dealTimesEachCore[i] == 0) {
                dealTimesEachCore[i] = 1;
            }
            dealLenEachTimeAttachTail += minEleRepeatTail;
        }
        endOffset[i] = startOffset[i] + eleLenEachCore;
        eleTotalEachCore[i] = eleLenEachCore;

        // é»è®¤å°±ç³è¯·è¿ä¹å¤§
        reduceMaxRstsLenEachCore[i] = DEAL_TIMES_EACH_CORE_REDUCE * ELEMENTS_IN_BLOCK + ELEMENTS_IN_BLOCK;
        dealLenUpBlockEachTime[i] = CeilA2B(dealLenEachTimeAttachTail, ELEMENTS_IN_BLOCK) * ELEMENTS_IN_BLOCK;

        totalRptCntNor[i] = dealLenEachTime[i] / elementsPerRepeat;
        totalRptCntNorRemainder[i] = dealLenEachTime[i] % elementsPerRepeat; // should calc
        rptBatchCntNor[i] = totalRptCntNor[i] / MAX_REPEATS;                 // limit by L0 API, should calc
        rptBatchCntNorRemainder[i] = totalRptCntNor[i] % MAX_REPEATS;        // should calc
        rmdRptLenNor[i] = rptBatchCntNorRemainder[i] * elementsPerRepeat;
    }
    uint32_t maxRepeatLen = MAX_REPEATS * elementsPerRepeat;

    // Fill tiling data structure
    tilingData.incx = incx;
    tilingData.needVecCoreNum = needVecCoreNum;
    tilingData.dtypeFlag = dtypeFlag;
    tilingData.rstLenAllCoreBytes = rstLenAllCoreBytes;
    tilingData.tailCount = minEleRepeatTail;
    tilingData.maxRepeatLen = maxRepeatLen;

    uint32_t copyLen = MAXVECTORNUM * sizeof(uint32_t);
    memcpy(tilingData.startOffset, startOffset, copyLen);
    memcpy(tilingData.eleTotalEachCore, eleTotalEachCore, copyLen);
    memcpy(tilingData.dealTimesEachCore, dealTimesEachCore, copyLen);
    memcpy(tilingData.dealLenEachTime, dealLenEachTime, copyLen);
    memcpy(tilingData.reduceMaxRstsLenEachCore, reduceMaxRstsLenEachCore, copyLen);
    memcpy(tilingData.dealLenUpBlockEachTime, dealLenUpBlockEachTime, copyLen);
    memcpy(tilingData.totalRptCntNor, totalRptCntNor, copyLen);
    memcpy(tilingData.totalRptCntNorRemainder, totalRptCntNorRemainder, copyLen);
    memcpy(tilingData.rptBatchCntNor, rptBatchCntNor, copyLen);
    memcpy(tilingData.rptBatchCntNorRemainder, rptBatchCntNorRemainder, copyLen);
    memcpy(tilingData.rmdRptLenNor, rmdRptLenNor, copyLen);

    delete[] allMem;

    return tilingData;
}

aclblasStatus_t aclblasIamax(aclblasHandle_t handle, const int64_t n, uint8_t* x, const int64_t incx, uint8_t* result)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream useStream = h->stream;

    uint32_t needVecCoreNum = 1;
    uint32_t mixCoreNum = CeilA2B(needVecCoreNum, 2);
    if (mixCoreNum == 0) {
        mixCoreNum = 1;
    }
    uint32_t numBlocks = mixCoreNum;

    uint32_t dtypeFlag = 0;
    uint32_t actualN = n;

    IamaxTilingData tiling = CalIamaxTilingData(n, incx, numBlocks, dtypeFlag);

    numBlocks = CeilA2B(tiling.needVecCoreNum, 2);
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    uint8_t* workspaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;
    size_t workspaceSize = 16 * 1024 * 1024 + MAXVECTORNUM * GM_RESULT_LEN * BYTE_LEN_4;

    aclError aclRet = aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(IamaxTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(workspaceDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(IamaxTilingData), &tiling, sizeof(IamaxTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    iamax_kernel_do(x, result, workspaceDevice, tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workspaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(workspaceDevice);
    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}
