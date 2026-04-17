# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

# ========= 配置不同 SOC 平台排除的算子 =========
# 定义 ascend950 平台不参与编译的算子列表
set(ASCEND950_EXCLUDED_OPS
    "dot"      # 950 编译排除
    "colwise_mul"
    "complex_mat_dot"
    "iamax"
    "nrm2"
    "sasum"
    "rot"
    "cgemv_batched"
    "cgerc"
    CACHE INTERNAL "Operators excluded for ascend950 platform"
)

# 根据当前 SOC 版本确定要排除的算子列表
set(EXCLUDED_OPS "")
string(TOLOWER "${SOC_VERSION}" SOC_VERSION_LOWER)
if(SOC_VERSION_LOWER MATCHES "^ascend950")
    set(EXCLUDED_OPS ${ASCEND950_EXCLUDED_OPS})
    if(EXCLUDED_OPS)
        message(STATUS "Excluded operators for ascend950: ${EXCLUDED_OPS}")
    endif()
endif()

# 收集当前目录下的基础源文件
file(GLOB BASE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# 自动收集所有算子目录的源文件（支持子目录中的架构特定目录）
set(OP_SRC_FILES "")
file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(child ${children})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child})
        # 检查该算子是否在排除列表中
        set(is_excluded FALSE)
        foreach(excluded_op ${EXCLUDED_OPS})
            if(child STREQUAL excluded_op)
                set(is_excluded TRUE)
                message(STATUS "Skipping operator '${child}' for ${SOC_VERSION}")
                break()
            endif()
        endforeach()
        
        # 如果算子未被排除，则收集其源文件
        if(NOT is_excluded)
            # 递归收集非架构特定源码，避免与 archXX 目录重复收集
            file(GLOB_RECURSE dir_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${child}/*.cpp)
            foreach(src_file ${dir_srcs})
                set(is_arch_specific FALSE)
                foreach(arch_dir ${ARCH_SPECIFIC_DIRS})
                    if(src_file MATCHES "/${arch_dir}/")
                        set(is_arch_specific TRUE)
                        break()
                    endif()
                endforeach()
                if(NOT is_arch_specific)
                    list(APPEND OP_SRC_FILES ${src_file})
                endif()
            endforeach()
        endif()
    endif()
endforeach()

# 收集子目录中的架构特定源文件
set(ARCH_SRC_FILES "")
foreach(arch_dir ${SOC_ARCH_DIRS})
    file(GLOB_RECURSE arch_dir_srcs
        ${CMAKE_CURRENT_SOURCE_DIR}/*/${arch_dir}/**/*.cpp
    )
    list(APPEND ARCH_SRC_FILES ${arch_dir_srcs})
endforeach()

set(ALL_BLAS_SRC_FILES
    ${BASE_SRC_FILES}
    ${OP_SRC_FILES}
    ${ARCH_SRC_FILES}
    PARENT_SCOPE
)