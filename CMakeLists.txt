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

project(ops_blas VERSION 1.0.0)

# ========= 基本配置 =========
if(NOT DEFINED SOC_VERSION OR SOC_VERSION STREQUAL "")
  set(SOC_VERSION ascend910b3)
endif()
string(TOLOWER "${SOC_VERSION}" SOC_VERSION_LOWER)
# SOC_VERSION -> NPU_ARCH 映射（ascend910b->dav-2201, ascend950->dav-3101）
if(SOC_VERSION_LOWER MATCHES "^ascend910b")
  set(NPU_ARCH "dav-2201")
elseif(SOC_VERSION_LOWER MATCHES "^ascend910_93")
  set(NPU_ARCH "dav-2201")
elseif(SOC_VERSION_LOWER MATCHES "^ascend950")
  set(NPU_ARCH "dav-3510")
elseif(SOC_VERSION_LOWER MATCHES "^ascend310p")
  set(NPU_ARCH "dav-1101")
else()
  message(FATAL_ERROR "Unsupported SOC_VERSION: ${SOC_VERSION}. Supported: ascend910b*, ascend910_93*, ascend950*, ascend310p*")
endif()
message(STATUS "SOC_VERSION=${SOC_VERSION}, NPU_ARCH=${NPU_ARCH}")

# ========= SOC 架构目录配置 =========
# 根据 SOC 返回需要编译的架构目录列表
function(get_soc_arch_dirs soc_version arch_dirs)
  string(TOLOWER "${soc_version}" soc_lower)
  set(dirs "")

  if(soc_lower MATCHES "^ascend910b")
    list(APPEND dirs "arch22")
  elseif(soc_lower MATCHES "^ascend910_93")
    list(APPEND dirs "arch22")
  elseif(soc_lower MATCHES "^ascend950")
    list(APPEND dirs "arch35")
  elseif(soc_lower MATCHES "^ascend310p")
    list(APPEND dirs "arch20")
  endif()

  set(${arch_dirs} ${dirs} PARENT_SCOPE)
endfunction()

# 获取当前 SOC 对应的架构目录
get_soc_arch_dirs(${SOC_VERSION} SOC_ARCH_DIRS)
message(STATUS "SOC_ARCH_DIRS=${SOC_ARCH_DIRS}")

# 定义所有支持的架构特定目录（全局唯一定义）
set(ARCH_SPECIFIC_DIRS arch35 arch22 arch20 CACHE INTERNAL "Architecture specific directories")

set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})

include(cmake/asc_devkit_version.cmake)
ops_blas_detect_asc_devkit_version()

set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "install path" FORCE)

# 外部传参
option(ENABLE_PACKAGE "Enable build package" OFF)

# 引入 ops-tensor
include(cmake/third_party/ops-tensor.cmake)

# ========= 构建 blasLt 动态库 =========
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
    ${CMAKE_CURRENT_LIST_DIR}/blasLt/utils
    ${CMAKE_CURRENT_LIST_DIR}/blasLt/include/kernel
    ${CMAKE_CURRENT_LIST_DIR}/blasLt/include/tiling
    ${OPTENSOR_INCLUDE_DIR}
    "${OPTENSOR_INCLUDE_DIR}/tensor_api"
)
set(_OPS_BLAS_ASC_ROOT "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc")
if(EXISTS "${_OPS_BLAS_ASC_ROOT}/include")
  target_include_directories(${OPS_BLASLT} PRIVATE
    "${_OPS_BLAS_ASC_ROOT}"
    "${_OPS_BLAS_ASC_ROOT}/include"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
  )
endif()

if(ENABLE_BLASLT_MXFP8)
  set_source_files_properties(
    ${CMAKE_CURRENT_LIST_DIR}/blasLt/matmul_mxfp8/arch35/matmul_mxfp8_host.cpp
    PROPERTIES LANGUAGE CXX)
endif()
target_compile_features(${OPS_BLASLT} PRIVATE cxx_std_17)

set(ASC_WARN_SUPPRESS "-Wno-#pragma-messages" "-Wno-deprecated-declarations")

target_compile_options(${OPS_BLASLT} PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH} ${ASC_WARN_SUPPRESS}>
)

target_link_directories(${OPS_BLASLT} PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(
  ${OPS_BLASLT}
  PRIVATE ascendc_runtime ascendcl runtime register error_manager profapi ge_common_base unified_dlog mmpa dl
          ascend_dump c_sec)

# ========= 构建 blas 动态库 =========
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
    ${ASCEND_CANN_PACKAGE_PATH}/include/op_common/
    ./include
    ${CMAKE_CURRENT_LIST_DIR}/blas
)

target_compile_options(${OPS_BLAS} PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH} ${ASC_WARN_SUPPRESS}>
)

# ========= 安装规则（CPack 打包时放入 lib64 和 include） =========
install(TARGETS ${OPS_BLASLT} ${OPS_BLAS}
    LIBRARY DESTINATION lib64
)
install(FILES
    include/cann_ops_blas.h
    include/cann_ops_blasLt.h
    include/cann_ops_blas_common.h
    DESTINATION include
)

# ========= 构建测试程序 =========
option(BUILD_TEST "Build test programs" OFF)
if(BUILD_TEST)
    add_subdirectory(test)
endif()

if(ENABLE_PACKAGE)
    include(cmake/package.cmake)
    pack(${SOC_VERSION})
endif()
