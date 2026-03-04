# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

add_executable(blasLtMatmul_test
    blasLtMatmul_test.cpp
)

target_include_directories(blasLtMatmul_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(blasLtMatmul_test PRIVATE
    ASCENDC_DUMP=0
)
target_compile_options(blasLtMatmul_test PRIVATE
    -g
)
target_include_directories(blasLtMatmul_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{LINUX_INCLUDE_PATH}
)

target_link_libraries(blasLtMatmul_test PRIVATE
    ${OPS_BLASLT}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
)