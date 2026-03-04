/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <vector>
#include <random>

#include "acl/acl.h"
#include "cann_ops_blasLt.h"
#include "../common/util.h"

#define GM_ADDR uint8_t*

int Init(int32_t deviceId, aclrtStream* stream)
{
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

aclError aclblasLtMatmulTest(int32_t deviceId, aclrtStream& stream)
{
  // 1. 定义问题参数
  // D = A (m x k) * B (k x n)
  const int m = 256, n = 256, k = 128;
  // D = alpha * A * B + beta * C
  const float alpha = 1.0f, beta = 0.0f;

  // 2. 创建 BLASLt 句柄
  aclblasLtHandle_t ltHandle;
  CHECK_ACLBLASLT(aclblasLtCreate(&ltHandle));

  // 3. 在Device上分配和初始化数据 (示例使用随机初始化)
  std::vector<float> hostInput(m * k, 0);
  std::vector<float> hostWeight(k * n, 0);
  std::vector<float> hostOutput(m * n, 0);
  std::vector<float> goldenOutput(m * n, 0);
  FillRandomData<float>(hostInput, -2.0f, 2.0f);
  FillRandomData<float>(hostWeight, -2.0f, 2.0f);
  // float *d_A, *d_B, *d_D;
  GM_ADDR d_A = nullptr;
  GM_ADDR d_B = nullptr;
  GM_ADDR d_D = nullptr;
  auto sizeInput = hostInput.size() * sizeof(float);
  auto sizeWeight = hostWeight.size() * sizeof(float);
  auto sizeOutput = hostOutput.size() * sizeof(float);
  CHECK_ACLRT(aclrtMalloc((void **)&d_A, sizeInput, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACLRT(aclrtMalloc((void **)&d_B, sizeWeight, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACLRT(aclrtMalloc((void **)&d_D, sizeOutput, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACLRT(aclrtMemcpy(d_A, sizeInput, hostInput.data(), sizeInput, ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACLRT(aclrtMemcpy(d_B, sizeWeight, hostWeight.data(), sizeWeight, ACL_MEMCPY_HOST_TO_DEVICE));

  // 4. 创建并设置矩阵布局描述符
  aclblasLtMatrixLayout_t Adesc, Bdesc, Ddesc;
  aclblasLtOrder_t order = ACLBLASLT_ORDER_ROW;
  // 矩阵 A: FP32, 行主序, 维度 m x k
  CHECK_ACLBLASLT(aclblasLtMatrixLayoutCreate(&Adesc, ACL_FLOAT, m, k, m));
  CHECK_ACLBLASLT(aclblasLtMatrixLayoutSetAttribute(Adesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int)));
  // 矩阵 B: FP32, 行主序, 维度 k x n
  CHECK_ACLBLASLT(aclblasLtMatrixLayoutCreate(&Bdesc, ACL_FLOAT, k, n, k));
  CHECK_ACLBLASLT(aclblasLtMatrixLayoutSetAttribute(Bdesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int)));
  // 矩阵 D: FP32, 行主序, 维度 m x n
  CHECK_ACLBLASLT(aclblasLtMatrixLayoutCreate(&Ddesc, ACL_FLOAT, m, n, m));
  CHECK_ACLBLASLT(aclblasLtMatrixLayoutSetAttribute(Ddesc, ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(int)));

  // 5. 创建并设置计算描述符
  aclblasLtMatmulDesc_t operationDesc;
  CHECK_ACLBLASLT(aclblasLtMatmulDescCreate(&operationDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT));
  // 设置 epilogue
  aclblasLtEpilogue_t epilogue = ACLBLASLT_EPILOGUE_DEFAULT;
  CHECK_ACLBLASLT(
      aclblasLtMatmulDescSetAttribute(operationDesc, ACLBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

  // 6. 算法选择
  aclblasLtMatmulPreference_t preference;
  CHECK_ACLBLASLT(aclblasLtMatmulPreferenceCreate(&preference));
  // 为算法最多预留 12MB 的临时工作空间
  size_t max_workspaceSize = 12 * 1024 * 1024;
  CHECK_ACLBLASLT(aclblasLtMatmulPreferenceSetAttribute(preference, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &max_workspaceSize, sizeof(max_workspaceSize)));

  // 用于接收推荐算法
  int returnedAlgoCount = 0;
  const int request_solutions = 1;
  aclblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
  CHECK_ACLBLASLT(aclblasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Ddesc, Ddesc, preference,
                                                  request_solutions, heuristicResult, &returnedAlgoCount));

  if (returnedAlgoCount == 0) {
    std::cerr << "No valid algorithm found for the given problem!" << std::endl;
    return -1;
  }
  // 选择第一个启发式推荐的算法
  const auto &algo = heuristicResult[0].algo;

  // 7. 分配工作空间并执行计算
  uint64_t workspace_size = 0;
  for(int i = 0; i < returnedAlgoCount; i++) {
    workspace_size = std::max(workspace_size, heuristicResult[i].workspaceSize);
  }
  void *d_workspace = nullptr;
  CHECK_ACLRT(aclrtMalloc(&d_workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACLBLASLT(aclblasLtMatmul(ltHandle,
                                  operationDesc,
                                  &alpha,
                                  d_A,
                                  Adesc,
                                  d_B,
                                  Bdesc,
                                  &beta,
                                  d_D,
                                  Ddesc,
                                  d_D,
                                  Ddesc,
                                  &algo,
                                  d_workspace,
                                  workspace_size,
                                  stream));

  // 8. 同步并检查错误
  auto ret = aclInit(nullptr);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  std::cout << "aclblasLtMatmul executed successfully!" << std::endl;
  std::cout << "Heuristic returned " << returnedAlgoCount << " algorithm(s). Using the first one." << std::endl;
  // 输出数据Device To Host
  ret = aclrtMemcpy(hostOutput.data(), sizeOutput, d_D, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy deviceOutput failed. ERROR: %d\n", ret); return ret);

  // 9. 计算golden，对比精度
  ComputeGolden<float>(m, k, n, hostInput, hostWeight, goldenOutput);
  std::cout << "start comparing result with golden value." << std::endl;
  std::vector<uint64_t> errorIndices = Compare<float>(hostOutput, goldenOutput);
  if (errorIndices.size() == 0) {
      std::cout << "matmul run successfully!" << std::endl;
  } else {
      for (uint64_t i : errorIndices) {
          uint64_t errIdx = errorIndices[i];
          std::cout << "error index: " << errIdx << ", output: " << hostOutput[errIdx]
                    << ", golden: " << goldenOutput[errIdx] << std::endl;
      }
      std::cout << "matmul run failed!" << std::endl;
  }

  // 10. 释放资源
  CHECK_ACLRT(aclrtFree(d_workspace));
  CHECK_ACLRT(aclrtFree(d_D));
  CHECK_ACLRT(aclrtFree(d_B));
  CHECK_ACLRT(aclrtFree(d_A));
  CHECK_ACLBLASLT(aclblasLtMatmulPreferenceDestroy(preference));
  CHECK_ACLBLASLT(aclblasLtMatmulDescDestroy(operationDesc));
  CHECK_ACLBLASLT(aclblasLtMatrixLayoutDestroy(Ddesc));
  CHECK_ACLBLASLT(aclblasLtMatrixLayoutDestroy(Bdesc));
  CHECK_ACLBLASLT(aclblasLtMatrixLayoutDestroy(Adesc));
  CHECK_ACLBLASLT(aclblasLtDestroy(ltHandle));

  return ACL_SUCCESS;
}

int main(int argc, char* argv[])
{
  // 获取输入Shape
  // int m, k, n;
  // try {
  //     parseArguments(argc, argv, m, k, n);
  // } catch (const std::exception& e) {
  //     std::cerr << e.what() << std::endl;
  //     printUsage(argv[0]);
  //     return 1;
  // }
  // 固定写法，device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  ret = aclblasLtMatmulTest(deviceId, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasLtMatmulTest failed. ERROR: %d\n", ret); return ret);
  // 固定写法，释放资源
  Finalize(deviceId, stream);
  return 0;
}