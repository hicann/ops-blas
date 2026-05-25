# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

add_executable(strttp_test arch35/strttp_test.cpp)

find_path(GTEST_INCLUDE_DIR gtest/gtest.h PATHS /usr/local/include /usr/include)
find_library(GTEST_LIB gtest PATHS /usr/local/lib /usr/lib)
find_library(GTEST_MAIN_LIB gtest_main PATHS /usr/local/lib /usr/lib)

target_include_directories(strttp_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{LINUX_INCLUDE_PATH}
    ${GTEST_INCLUDE_DIR}
)

target_link_libraries(strttp_test PRIVATE
    ${OPS_BLAS}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
    ${GTEST_LIB}
    ${GTEST_MAIN_LIB}
    pthread
)

ops_blas_add_tests(${OPS_BLAS})
