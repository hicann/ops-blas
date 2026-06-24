# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

ops_blas_add_gtest_tests(${OPS_BLAS} gemm_batched_ex_test)

# Reuse parseAclDataType / physRows / physCols from gemm_ex
target_include_directories(gemm_batched_ex_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../gemm/gemm_ex)

# gemm uses matmul_tiling::MultiCoreMatmulTiling which requires tiling_api (static) and
# platform (shared) symbols. Since tiling_api is a static library, its symbols are not
# transitively available from libops_blas.so, so we need to link them explicitly.
target_link_directories(gemm_batched_ex_test PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
target_link_libraries(gemm_batched_ex_test PRIVATE tiling_api platform c_sec)
