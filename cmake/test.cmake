# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# 检查 blas/<op_name>/ 下是否有当前 SOC 可编译的算子实现
function(_ops_blas_has_blas_op_sources op_name out_var)
    set(has_sources FALSE)
    foreach(arch_dir ${SOC_ARCH_DIRS})
        file(GLOB arch_dir_srcs ${CMAKE_SOURCE_DIR}/blas/${op_name}/${arch_dir}/*.cpp
                                  ${CMAKE_SOURCE_DIR}/blas/*/${op_name}/${arch_dir}/*.cpp)
        if(arch_dir_srcs)
            set(has_sources TRUE)
            break()
        endif()
    endforeach()
    if(NOT has_sources)
        file(GLOB base_srcs ${CMAKE_SOURCE_DIR}/blas/${op_name}/*.cpp
                              ${CMAKE_SOURCE_DIR}/blas/*/${op_name}/*.cpp)
        if(base_srcs)
            set(has_sources TRUE)
        endif()
    endif()
    set(${out_var} ${has_sources} PARENT_SCOPE)
endfunction()

# 检查 blasLt/ 下是否有当前 SOC 可编译的算子实现。
# 测试名如 blasLtMatmul 会匹配 blasLt 下名称包含 matmul 的子目录（如 matmul_fp32、matmul_mxfp8）。
function(_ops_blas_has_blaslt_op_sources test_name out_var)
    set(has_sources FALSE)
    if(NOT test_name MATCHES "^blasLt")
        set(${out_var} FALSE PARENT_SCOPE)
        return()
    endif()

    string(REGEX REPLACE "^blasLt" "" _op_suffix ${test_name})
    string(TOLOWER "${_op_suffix}" _op_suffix_lower)

    file(GLOB children RELATIVE ${CMAKE_SOURCE_DIR}/blasLt ${CMAKE_SOURCE_DIR}/blasLt/*)
    foreach(child ${children})
        if(NOT IS_DIRECTORY ${CMAKE_SOURCE_DIR}/blasLt/${child})
            continue()
        endif()
        if(child STREQUAL "include" OR child STREQUAL "utils")
            continue()
        endif()
        string(TOLOWER "${child}" _child_lower)
        if(NOT _child_lower MATCHES "${_op_suffix_lower}")
            continue()
        endif()

        file(GLOB_RECURSE dir_srcs ${CMAKE_SOURCE_DIR}/blasLt/${child}/*.cpp)
        foreach(src_file ${dir_srcs})
            set(is_arch_specific FALSE)
            foreach(arch_dir ${ARCH_SPECIFIC_DIRS})
                if(src_file MATCHES "/${arch_dir}/")
                    set(is_arch_specific TRUE)
                    break()
                endif()
            endforeach()
            if(NOT is_arch_specific)
                set(has_sources TRUE)
                break()
            endif()
        endforeach()
        if(has_sources)
            break()
        endif()

        foreach(arch_dir ${SOC_ARCH_DIRS})
            file(GLOB arch_dir_srcs ${CMAKE_SOURCE_DIR}/blasLt/${child}/${arch_dir}/*.cpp)
            if(arch_dir_srcs)
                if((child STREQUAL "matmul_mxfp8" OR child STREQUAL "matmul_mxfp4") AND NOT ENABLE_BLASLT_MXFP8)
                    continue()
                endif()
                set(has_sources TRUE)
                break()
            endif()
        endforeach()
        if(has_sources)
            break()
        endif()
    endforeach()

    set(${out_var} ${has_sources} PARENT_SCOPE)
endfunction()

# 检查算子在当前 SOC 下是否有可编译实现（同时覆盖 blas/ 与 blasLt/ 两种目录布局）
function(ops_blas_has_op_sources_for_soc test_name out_var)
    _ops_blas_has_blas_op_sources(${test_name} _blas_has)
    if(_blas_has)
        set(${out_var} TRUE PARENT_SCOPE)
        return()
    endif()
    _ops_blas_has_blaslt_op_sources(${test_name} _blaslt_has)
    set(${out_var} ${_blaslt_has} PARENT_SCOPE)
endfunction()

# 为指定测试目标收集源文件：根目录 ${target}.cpp + 当前 SOC 架构目录下的同名/同前缀补充源
function(ops_blas_get_test_target_sources target out_var)
    set(sources "")
    set(root_src ${CMAKE_CURRENT_SOURCE_DIR}/${target}.cpp)

    # Check if any arch directory has a replacement for the root test
    set(_has_arch_test FALSE)
    foreach(arch_dir ${SOC_ARCH_DIRS})
        set(arch_src ${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/${target}.cpp)
        if(EXISTS ${arch_src})
            list(APPEND sources ${arch_src})
            set(_has_arch_test TRUE)
        endif()
        file(GLOB arch_supp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/${target}_*.cpp)
        list(APPEND sources ${arch_supp_srcs})
    endforeach()

    # Only include root test if no arch-specific replacement exists.
    # Root tests may use old API conventions (e.g. host pointers) that are
    # incompatible with arch-specific host implementations (e.g. device pointers).
    if(NOT _has_arch_test AND EXISTS ${root_src})
        list(APPEND sources ${root_src})
    endif()

    if(NOT sources)
        message(FATAL_ERROR "No test sources found for target '${target}' in ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    set(${out_var} ${sources} PARENT_SCOPE)
endfunction()

function(_ops_blas_register_test_target target link_lib)
    _ops_blas_ensure_refblas_found()
    ops_blas_get_test_target_sources(${target} _test_srcs)
    add_executable(${target} ${_test_srcs})

    if(DEFINED TEST_DEVICE_ID)
        target_compile_definitions(${target} PRIVATE TEST_DEVICE_ID=${TEST_DEVICE_ID})
    endif()

    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/test/frame
        ${CMAKE_SOURCE_DIR}/test/utils
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/blas/common/helper
        $ENV{LINUX_INCLUDE_PATH}
        ${REFBLAS_INCLUDE_DIR}
    )
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_link_libraries(${target} PRIVATE
        ${link_lib}
        $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
        ${REFBLAS_LIB}
        ${REFLAPACK_LIB}
    )

    _ops_blas_copy_test_config_files(${target})
endfunction()

# Locate GTest once per configure (CSV-driven ST uses custom main, not gtest_main).
function(_ops_blas_ensure_gtest_found)
    if(NOT GTEST_LIB OR NOT GTEST_INCLUDE_DIR)
        find_path(GTEST_INCLUDE_DIR gtest/gtest.h PATHS /usr/local/include /usr/include)
        find_library(GTEST_LIB gtest PATHS /usr/local/lib /usr/lib)
    endif()
endfunction()

# Locate reference BLAS (CBLAS) and LAPACK once per configure.
function(_ops_blas_ensure_refblas_found)
    if(NOT REFBLAS_LIB)
        find_path(REFBLAS_INCLUDE_DIR cblas.h
            PATHS /usr/local/include /usr/include /usr/include/x86_64-linux-gnu
                  ${HOMEBREW_PREFIX}/include)
        find_library(REFBLAS_LIB NAMES blas
            PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
                  ${HOMEBREW_PREFIX}/lib
            PATH_SUFFIXES blas)
        find_library(REFLAPACK_LIB NAMES lapack
            PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
                  ${HOMEBREW_PREFIX}/lib
            PATH_SUFFIXES lapack)
        if(NOT REFBLAS_INCLUDE_DIR OR NOT REFBLAS_LIB)
            message(FATAL_ERROR "Reference BLAS (cblas.h / libblas) not found. "
                "Install via: apt-get install libblas-dev")
        endif()
        if(NOT REFLAPACK_LIB)
            message(WARNING "Reference LAPACK (liblapack) not found. "
                "LAPACK-based golden tests will fail to link.")
        endif()
    endif()
endfunction()

# Copy CSV/JSON from test root and from current-SOC arch subdirs (e.g. arch35/).
function(_ops_blas_copy_test_config_files target)
    set(_cfg_files "")
    file(GLOB _root_json "${CMAKE_CURRENT_SOURCE_DIR}/*.json")
    file(GLOB _root_csv "${CMAKE_CURRENT_SOURCE_DIR}/*.csv")
    list(APPEND _cfg_files ${_root_json} ${_root_csv})

    foreach(arch_dir ${SOC_ARCH_DIRS})
        file(GLOB _arch_json "${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/*.json")
        file(GLOB _arch_csv "${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/*.csv")
        list(APPEND _cfg_files ${_arch_json} ${_arch_csv})
    endforeach()

    # Legacy single-file JSON under build/json_configs/
    set(_json_src "${CMAKE_BINARY_DIR}/json_configs/${target}_testcases.json")
    set(_json_src_alt "${CMAKE_SOURCE_DIR}/build/test/json_configs/${target}_testcases.json")
    if(EXISTS ${_json_src})
        list(APPEND _cfg_files ${_json_src})
    elseif(EXISTS ${_json_src_alt})
        list(APPEND _cfg_files ${_json_src_alt})
    endif()

    foreach(_cfg_file ${_cfg_files})
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${_cfg_file}
                    $<TARGET_FILE_DIR:${target}>)
    endforeach()
endfunction()

# Register GTest-based CSV/JSON ST target (custom main; links gtest only, not gtest_main).
function(_ops_blas_register_gtest_target target link_lib)
    _ops_blas_ensure_gtest_found()
    _ops_blas_ensure_refblas_found()
    ops_blas_get_test_target_sources(${target} _test_srcs)
    list(APPEND _test_srcs ${CMAKE_SOURCE_DIR}/test/frame/test_main.cpp)
    add_executable(${target} ${_test_srcs})

    if(DEFINED TEST_DEVICE_ID)
        target_compile_definitions(${target} PRIVATE TEST_DEVICE_ID=${TEST_DEVICE_ID})
    endif()

    set(_extra_includes "")
    foreach(arch_dir ${SOC_ARCH_DIRS})
        if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}")
            list(APPEND _extra_includes "${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}")
        endif()
    endforeach()

    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/test/frame
        ${CMAKE_SOURCE_DIR}/test/utils
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${_extra_includes}
        $ENV{LINUX_INCLUDE_PATH}
        ${GTEST_INCLUDE_DIR}
        ${REFBLAS_INCLUDE_DIR}
    )
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_link_libraries(${target} PRIVATE
        ${link_lib}
        $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
        ${GTEST_LIB}
        ${REFBLAS_LIB}
        ${REFLAPACK_LIB}
        pthread
    )

    _ops_blas_copy_test_config_files(${target})
endfunction()

function(_ops_blas_discover_test_targets out_var)
    set(targets "")

    file(GLOB root_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    foreach(src ${root_srcs})
        get_filename_component(target ${src} NAME_WE)
        list(APPEND targets ${target})
    endforeach()

    foreach(arch_dir ${SOC_ARCH_DIRS})
        file(GLOB arch_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/*.cpp)
        foreach(src ${arch_srcs})
            get_filename_component(target ${src} NAME_WE)
            if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${target}.cpp)
                continue()
            endif()
            list(APPEND targets ${target})
        endforeach()
    endforeach()

    if(targets)
        list(REMOVE_DUPLICATES targets)
    endif()
    set(${out_var} ${targets} PARENT_SCOPE)
endfunction()

# 注册测试可执行文件。ARGN 为空时自动发现本目录 target；否则仅注册指定 target 列表。
function(ops_blas_add_tests link_lib)
    if(ARGN)
        set(targets ${ARGN})
    else()
        _ops_blas_discover_test_targets(targets)
    endif()

    if(NOT targets)
        message(FATAL_ERROR "No test targets to register in ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    foreach(target ${targets})
        if(TARGET ${target})
            continue()
        endif()
        _ops_blas_register_test_target(${target} ${link_lib})
    endforeach()
endfunction()

# Register GTest CSV/JSON ST targets. ARGN empty -> auto-discover *_test.cpp in root/arch dirs.
function(ops_blas_add_gtest_tests link_lib)
    if(ARGN)
        set(targets ${ARGN})
    else()
        _ops_blas_discover_test_targets(targets)
    endif()

    if(NOT targets)
        message(FATAL_ERROR "No GTest test targets to register in ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    foreach(target ${targets})
        if(TARGET ${target})
            continue()
        endif()
        _ops_blas_register_gtest_target(${target} ${link_lib})
    endforeach()
endfunction()
