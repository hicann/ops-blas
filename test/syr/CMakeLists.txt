# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# Determine which test source and executable to build based on current SOC architecture
if("arch35" IN_LIST SOC_ARCH_DIRS)
    set(SYR_TEST_SRC syr_test.cpp)
elseif("arch22" IN_LIST SOC_ARCH_DIRS)
    set(SYR_TEST_SRC syr_a2_test.cpp)
else()
    set(SYR_TEST_SRC syr_test.cpp)
endif()

add_executable(syr_test
    ${SYR_TEST_SRC}
)

target_include_directories(syr_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{LINUX_INCLUDE_PATH}
)

target_link_libraries(syr_test PRIVATE
    ${OPS_BLAS}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
)
