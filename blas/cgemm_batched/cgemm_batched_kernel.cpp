/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "../common/common.h"
#include "../common/common_func.h"
#include "../common/iterator.h"
#include "../common/mma.h"
#include "complex_matmul_base_impl.h"

#ifdef __DAV_C220_VEC__
constexpr int64_t AIV_IN_GROUP_CORE_NUM = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t GATHER_NUM = 1024;
constexpr int64_t MAX_PROCESS_SIZE = (192 * 1024 - GATHER_NUM * sizeof(uint32_t)) / 2 / 2;
constexpr int64_t REPEAT_BYTES = 256;
#endif

__aicore__ __attribute__((always_inline)) int64_t inline ceil(int64_t x, int64_t base)
{
    return (x + base - 1) / base * base;
}

__aicore__ __attribute__((always_inline)) int64_t inline ceilDev(int64_t x, int64_t base)
{
    return (x + base - 1) / base;
}

#ifdef __DAV_C220_CUBE__
template <typename DTYPE = float>
class Conv1dAic {
public:
    __aicore__ __attribute__((always_inline)) inline Conv1dAic()
    {}

    __aicore__ __attribute__((always_inline)) inline void SetArgs(__gm__ uint8_t *__restrict__ aTensor,
        __gm__ uint8_t *__restrict__ bTensor, __gm__ uint8_t *__restrict__ cTensor,
        __gm__ uint8_t *__restrict__ workspace, __gm__ uint8_t *__restrict__ tiling)
    {
        AscendC::SetLoadDataPaddingValue<DTYPE>(0.0);
        AscendC::SetAtomicNone();
        AscendC::SetMaskNorm();
        AscendC::SetFixpipeNz2ndFlag(1, 0, 0);

        int64_t aiCoreNum = static_cast<int64_t>(AscendC::GetBlockNum());
        vecNumPerAiCore = AIV_IN_GROUP_CORE_NUM;
        blockIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());

        a_gm = reinterpret_cast<__gm__ DTYPE *>(aTensor);
        b_gm = reinterpret_cast<__gm__ DTYPE *>(bTensor);
        c_gm = reinterpret_cast<__gm__ DTYPE *>(cTensor);
        base_workspace_gm = reinterpret_cast<__gm__ DTYPE *>(workspace);

        a_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE *>(a_gm));
        b_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE *>(b_gm));
        c_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE *>(c_gm));
        base_workspace_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE *>(base_workspace_gm));

        M = (int64_t)(*((__gm__ int64_t *)tiling));
        K = (int64_t)(*((__gm__ int64_t *)tiling + 1)) * 2;
        N = (int64_t)(*((__gm__ int64_t *)tiling + 2)) * 2;
        batch = (int64_t)(*((__gm__ int64_t *)tiling + 3));
        subBatch = (int64_t)(*((__gm__ int64_t *)tiling + 4));
        miniBatch = (int64_t)(*((__gm__ int64_t *)tiling + 5));

        leftEles = M * K;
        rightEles = K * N / 2;
        resEles = M * N;

        int64_t bigCoreNum = (batch % aiCoreNum) == 0 ? aiCoreNum : (batch % aiCoreNum);

        processBatchAiCore = blockIdx < bigCoreNum ? subBatch : subBatch - 1;
        baseOffsetBatch = blockIdx < bigCoreNum ? blockIdx * subBatch
                                                : bigCoreNum * subBatch + (blockIdx - bigCoreNum) * (subBatch - 1);
        baseWorkspaceOffsetBatch = blockIdx * vecNumPerAiCore * miniBatch;

        int64_t workspacePingPongOffsetBatch = aiCoreNum * vecNumPerAiCore * miniBatch;
        workspace_gm_tensor_ping = base_workspace_gm_tensor[0];
        workspace_gm_tensor_pong = base_workspace_gm_tensor[workspacePingPongOffsetBatch * rightEles];
    }

    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        int64_t flagId = 0;
        for (int64_t i = 0; i < ceilDev(processBatchAiCore, vecNumPerAiCore * miniBatch); i++) {
            int64_t cvId = i / vecNumPerAiCore;

            AscendC::GlobalTensor<DTYPE> workspace_gm_tensor =
                (flagId == 0) ? workspace_gm_tensor_ping : workspace_gm_tensor_pong;

            int64_t start = i * vecNumPerAiCore * miniBatch;
            int64_t currentBatch = vecNumPerAiCore * miniBatch;
            if (start + vecNumPerAiCore * miniBatch > processBatchAiCore) {
                currentBatch = processBatchAiCore - start;
            }
            int64_t offsetBatch = baseOffsetBatch + start;

            AscendC::CrossCoreWaitFlag(0x8);

            batch_base_matmul_cfp32_dev(a_gm_tensor[offsetBatch * leftEles],
                b_gm_tensor[offsetBatch * rightEles],
                workspace_gm_tensor[baseWorkspaceOffsetBatch * rightEles],
                c_gm_tensor[offsetBatch * resEles],
                l1_base_a_tensor,
                l1_base_b_tensor,
                l0a_base_tensor,
                l0b_base_tensor,
                l0c_buf_tensor,
                currentBatch,
                M,
                K,
                N,
                0,
                0);

            if (cvId < (ceilDev(processBatchAiCore, vecNumPerAiCore * miniBatch) - 1)) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x7);
            }

            flagId = 1 - flagId;
        }
    }

private:
    __gm__ DTYPE *__restrict__ a_gm{nullptr};
    __gm__ DTYPE *__restrict__ b_gm{nullptr};
    __gm__ DTYPE *__restrict__ c_gm{nullptr};
    __gm__ DTYPE *__restrict__ base_workspace_gm{nullptr};

    uint64_t blockIdx{0};
    int64_t vecNumPerAiCore{0};

    AscendC::GlobalTensor<DTYPE> a_gm_tensor;
    AscendC::GlobalTensor<DTYPE> b_gm_tensor;
    AscendC::GlobalTensor<DTYPE> c_gm_tensor;
    AscendC::GlobalTensor<DTYPE> base_workspace_gm_tensor;
    AscendC::GlobalTensor<DTYPE> workspace_gm_tensor_ping;
    AscendC::GlobalTensor<DTYPE> workspace_gm_tensor_pong;

    const uint32_t l1_base_a_offset = 0;
    const uint32_t l1_base_b_offset = 128 * 1024;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<DTYPE> l1_base_a_tensor = buf.GetBuffer<BufferType::ASCEND_CB, DTYPE>(l1_base_a_offset);
    AscendC::LocalTensor<DTYPE> l1_base_b_tensor = buf.GetBuffer<BufferType::ASCEND_CB, DTYPE>(l1_base_b_offset);

    AscendC::LocalTensor<DTYPE> l0a_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, DTYPE>(0);
    AscendC::LocalTensor<DTYPE> l0b_base_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, DTYPE>(0);
    AscendC::LocalTensor<float> l0c_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    int64_t M{0};
    int64_t N{0};
    int64_t K{0};
    int64_t batch{0};
    int64_t subBatch{0};
    int64_t miniBatch{0};
    int64_t baseOffsetBatch{0};
    int64_t baseWorkspaceOffsetBatch{0};
    int64_t processBatchAiCore{0};
    int64_t leftEles{0};
    int64_t rightEles{0};
    int64_t resEles{0};
};
#endif

#ifdef __DAV_C220_VEC__
template <typename DTYPE = float>
class Conv1dAiv {
public:
    __aicore__ __attribute__((always_inline)) inline Conv1dAiv()
    {}

    __aicore__ __attribute__((always_inline)) inline void SetArgs(__gm__ uint8_t *__restrict__ aTensor,
        __gm__ uint8_t *__restrict__ bTensor, __gm__ uint8_t *__restrict__ gatherOffsetTensor,
        __gm__ uint8_t *__restrict__ cTensor, __gm__ uint8_t *__restrict__ workspace,
        __gm__ uint8_t *__restrict__ tiling)
    {
        AscendC::SetAtomicNone();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

        int64_t aiCoreNum = static_cast<int64_t>(AscendC::GetBlockNum());
        vecNumPerAiCore = static_cast<int64_t>(AscendC::GetSubBlockNum());
        blockIdx = block_idx;
        subBlockIdx = static_cast<int64_t>(AscendC::GetSubBlockIdx());
        int64_t vecIdx = static_cast<int64_t>(AscendC::GetBlockIdx());

        M = (int64_t)(*((__gm__ int64_t *)tiling));
        K = (int64_t)(*((__gm__ int64_t *)tiling + 1));
        N = (int64_t)(*((__gm__ int64_t *)tiling + 2));
        batch = (int64_t)(*((__gm__ int64_t *)tiling + 3));
        subBatch = (int64_t)(*((__gm__ int64_t *)tiling + 4));
        miniBatch = (int64_t)(*((__gm__ int64_t *)tiling + 5));

        rightEles = K * N * 2;

        int64_t bigCoreNum = (batch % aiCoreNum) == 0 ? aiCoreNum : (batch % aiCoreNum);

        processBatchAiCore = blockIdx < bigCoreNum ? subBatch : subBatch - 1;
        bigHalfBatch = ceilDev(processBatchAiCore, 2);
        smallHalfBatch = processBatchAiCore - bigHalfBatch;
        processBatch = bigHalfBatch * (1 - subBlockIdx) + smallHalfBatch * subBlockIdx;
        baseOffsetBatch = blockIdx < bigCoreNum ? blockIdx * subBatch
                                                : bigCoreNum * subBatch + (blockIdx - bigCoreNum) * (subBatch - 1);

        b_gm = reinterpret_cast<__gm__ DTYPE *>(bTensor);
        gather_offset_gm = reinterpret_cast<__gm__ uint32_t *>(gatherOffsetTensor);
        c_gm = reinterpret_cast<__gm__ DTYPE *>(cTensor);
        base_workspace_gm = reinterpret_cast<__gm__ DTYPE *>(workspace);

        b_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE *>(b_gm));
        gather_offset_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gather_offset_gm));
        c_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE *>(c_gm));
        base_workspace_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE *>(base_workspace_gm));

        int64_t oneBufferEles = miniBatch * rightEles;
        workspace_gm_tensor_ping = base_workspace_gm_tensor[blockIdx * vecNumPerAiCore * oneBufferEles];
        workspace_gm_tensor_pong = base_workspace_gm_tensor[aiCoreNum * vecNumPerAiCore * oneBufferEles +
                                                            blockIdx * vecNumPerAiCore * oneBufferEles];
    }

    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        AscendC::DataCopy(gather_offset_tensor, gather_offset_gm_tensor, GATHER_NUM);

        int64_t flagId = 0;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);

        for (int64_t i = 0; i < ceilDev(processBatchAiCore, vecNumPerAiCore * miniBatch); i++) {
            int64_t cvId = i / vecNumPerAiCore;

            int64_t start = vecNumPerAiCore * i * miniBatch;
            int64_t inOffset = baseOffsetBatch * rightEles + start * rightEles;
            int64_t outOffset = 0;

            int64_t currentOffset = subBlockIdx * miniBatch * rightEles;
            int64_t currentBatch = miniBatch;
            if (start + vecNumPerAiCore * miniBatch > processBatchAiCore) {
                int64_t currentBatchAiCore = processBatchAiCore - start;
                int64_t currentBatchBigHalf = ceilDev(currentBatchAiCore, vecNumPerAiCore);
                int64_t currentBatchSmallHalf = currentBatchAiCore - currentBatchBigHalf;

                currentOffset = subBlockIdx * currentBatchBigHalf * rightEles;
                currentBatch = (1 - subBlockIdx) * currentBatchBigHalf + subBlockIdx * currentBatchSmallHalf;
            }
            int64_t currEles = currentBatch * rightEles;

            if (currentBatch == 0) {
                if (cvId > 0) {
                    AscendC::CrossCoreWaitFlag(0x7);
                }
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8);
                break;
            }

            if (cvId > 0) {
                AscendC::CrossCoreWaitFlag(0x7);
            }

            event_t enentId = (flagId == 0) ? EVENT_ID0 : EVENT_ID1;
            AscendC::LocalTensor<DTYPE> raw_tensor = (flagId == 0) ? raw_tensor_ping : raw_tensor_pong;
            AscendC::LocalTensor<DTYPE> processed_tensor =
                (flagId == 0) ? processed_tensor_ping : processed_tensor_pong;
            AscendC::GlobalTensor<DTYPE> workspace_gm_tensor =
                (flagId == 0) ? workspace_gm_tensor_ping : workspace_gm_tensor_pong;

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(enentId);
            AscendC::DataCopyExtParams dataCopyExtParams{1, static_cast<uint32_t>(currEles * sizeof(DTYPE)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DTYPE> dataCopyInPadExtParams{false, 0, 0, 0};
            AscendC::DataCopyPad(
                raw_tensor, b_gm_tensor[inOffset + currentOffset], dataCopyExtParams, dataCopyInPadExtParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(enentId);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(enentId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(enentId);
            for (int64_t gatherStart = 0; gatherStart < currEles; gatherStart += GATHER_NUM) {
                int64_t num = GATHER_NUM;
                if (gatherStart + GATHER_NUM > currEles) {
                    num = currEles - gatherStart;
                }
                AscendC::Gather(
                    processed_tensor[gatherStart], raw_tensor[gatherStart], gather_offset_tensor, (uint32_t)0, num);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(enentId);

            AscendC::PipeBarrier<PIPE_V>();
            int64_t repeatTimes = ceilDev(currEles * sizeof(DTYPE), REPEAT_BYTES);
            AscendC::UnaryRepeatParams repeatParams{1, 1, 8, 8};

            AscendC::Muls(processed_tensor, processed_tensor, (DTYPE)(-1.0f), mask4Muls, repeatTimes, repeatParams);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(enentId);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(enentId);
            AscendC::DataCopyPad(workspace_gm_tensor[outOffset + currentOffset], processed_tensor, dataCopyExtParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(enentId);

            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8);

            flagId = 1 - flagId;
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }

private:
    uint64_t blockIdx{0};
    uint64_t subBlockIdx{0};
    uint64_t vecNumPerAiCore{0};

    __gm__ DTYPE *__restrict__ b_gm{nullptr};
    __gm__ uint32_t *__restrict__ gather_offset_gm{nullptr};
    __gm__ DTYPE *__restrict__ c_gm{nullptr};
    __gm__ DTYPE *__restrict__ base_workspace_gm{nullptr};

    AscendC::GlobalTensor<DTYPE> b_gm_tensor;
    AscendC::GlobalTensor<uint32_t> gather_offset_gm_tensor;
    AscendC::GlobalTensor<DTYPE> c_gm_tensor;
    AscendC::GlobalTensor<DTYPE> base_workspace_gm_tensor;
    AscendC::GlobalTensor<DTYPE> workspace_gm_tensor_ping;
    AscendC::GlobalTensor<DTYPE> workspace_gm_tensor_pong;

    int64_t M{0};
    int64_t N{0};
    int64_t K{0};
    int64_t batch{0};
    int64_t rightEles{0};
    int64_t subBatch{0};
    int64_t miniBatch{0};
    int64_t bigHalfBatch{0};
    int64_t smallHalfBatch{0};
    int64_t processBatchAiCore{0};
    int64_t processBatch{0};
    int64_t baseOffsetBatch{0};

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<uint32_t> gather_offset_tensor = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(0);
    AscendC::LocalTensor<DTYPE> raw_tensor_ping =
        buf.GetBuffer<BufferType::ASCEND_UB, DTYPE>(GATHER_NUM * sizeof(uint32_t));
    AscendC::LocalTensor<DTYPE> raw_tensor_pong =
        buf.GetBuffer<BufferType::ASCEND_UB, DTYPE>(GATHER_NUM * sizeof(uint32_t) + MAX_PROCESS_SIZE);
    AscendC::LocalTensor<DTYPE> processed_tensor_ping =
        buf.GetBuffer<BufferType::ASCEND_UB, DTYPE>(GATHER_NUM * sizeof(uint32_t) + MAX_PROCESS_SIZE * 2);
    AscendC::LocalTensor<DTYPE> processed_tensor_pong =
        buf.GetBuffer<BufferType::ASCEND_UB, DTYPE>(GATHER_NUM * sizeof(uint32_t) + MAX_PROCESS_SIZE * 3);

    uint64_t mask4Muls[2] = {0b0101010101010101010101010101010101010101010101010101010101010101,
        0b0101010101010101010101010101010101010101010101010101010101010101};
};
#endif

extern "C" __global__ __aicore__ void cgemm_batched(
    GM_ADDR a, GM_ADDR b, GM_ADDR gatherOffset, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();

#ifdef __DAV_C220_CUBE__
    Conv1dAic<float> conv1d_aic{};
    conv1d_aic.SetArgs(a, b, c, workspace, tiling);
    conv1d_aic.Run();
#endif
#ifdef __DAV_C220_VEC__
    Conv1dAiv<float> conv1d_aiv{};
    conv1d_aiv.SetArgs(a, b, gatherOffset, c, workspace, tiling);
    conv1d_aiv.Run();
#endif
}

void cgemm_batched_kernel_do(GM_ADDR a, GM_ADDR b, GM_ADDR gatherOffset, GM_ADDR c,
                              GM_ADDR workSpace, GM_ADDR tilingGm,
                              uint32_t numBlocks, void *stream)
{
    cgemm_batched<<<numBlocks, nullptr, stream>>>(a, b, gatherOffset, c, workSpace, tilingGm);
}