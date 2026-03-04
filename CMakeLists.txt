# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)

project(ops_blas)

# ========= 基本配置 =========
set(SOC_VERSION ascend910b3)
set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "install path" FORCE)

# 外部传参
option(ENABLE_PACKAGE "Enable build package" OFF)

# ========= 查找 ascendc cmake =========
# if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
#     set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
# elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
#     set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
# elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
#     set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
# else()
#     message(FATAL_ERROR "ascendc_kernel_cmake does not exist.")
# endif()

# include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ========= 构建动态库 =========
# blasLt
set(ALL_BLASLT_SRC_FILES "")
set(OPS_BLASLT ops_blasLt)
find_package(ASC REQUIRED)
add_subdirectory(blasLt)

project(${OPS_BLASLT} LANGUAGES ASC CXX)
add_library(${OPS_BLASLT} SHARED ${ALL_BLASLT_SRC_FILES})

set_source_files_properties(
    ${ALL_BLAS_SRC_FILES}
    PROPERTIES LANGUAGE ASC
)

set_source_files_properties(
    ${ALL_BLASLT_SRC_FILES}
    PROPERTIES LANGUAGE ASC
)

target_include_directories(${OPS_BLASLT} PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
    ./include
)

target_compile_options(${OPS_BLASLT} PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3101>
)

# blas
set(ALL_BLAS_SRC_FILES "")
set(OPS_BLAS ops_blas)

add_subdirectory(blas)

project(${OPS_BLAS} LANGUAGES ASC CXX)

add_library(${OPS_BLAS} SHARED ${ALL_BLAS_SRC_FILES})

set_source_files_properties(
    ${ALL_BLAS_SRC_FILES}
    PROPERTIES LANGUAGE ASC
)

target_include_directories(${OPS_BLAS} PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
    ./include
)

target_compile_options(${OPS_BLAS} PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3101>
)

# ========= 构建测试程序 =========
option(BUILD_TEST "Build test programs" OFF)
if(BUILD_TEST)
    add_subdirectory(test)
endif()

if(ENABLE_PACKAGE)
    include(cmake/package.cmake)
    pack()
endif()
