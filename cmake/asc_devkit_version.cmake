# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 读取 asc_devkit_version.h，判断是否满足所需版本：ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR > 0
# - MXFP8/MXFP4：blasLt 矩阵乘法，需 asc-devkit >= 9.1
# - TENSOR_API_OPS 列表中的算子：仅 arch35(ascend950) 使用 tensor_api，需 asc-devkit >= 9.1；
#   其他架构不受此版本限制。新增算子只需在下方列表中加一行。
# - TRSM_BLOCKED：仅 arch35(ascend950) 的 strsm_blocked_kernel 使用 tensor_api，需 asc-devkit >= 9.1；
#   strsm_kernel(SIMT path) 不依赖 tensor_api，低版本下 fallback 到 SIMT（特例，不纳入列表驱动）
function(ops_blas_detect_asc_devkit_version)
  set(_header
      "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/version/asc_devkit_version.h")
  set(ASC_DEVKIT_MAJOR 0)
  set(ASC_DEVKIT_MINOR 0)
  set(ENABLE_BLASLT_MXFP8 FALSE)

  # 列表驱动的 tensor_api 算子注册。
  # 格式: VAR_SUFFIX|blas_filter_regex|test_name1,test_name2
  # 使用 '|' 作为字段分隔符（CMake 列表以 ';' 分隔，字段内不能用 ';'）。
  # 新增算子只需在此列表中添加一行，cmake/asc_devkit_version.cmake、blas/CMakeLists.txt、
  # cmake/test.cmake、test/CMakeLists.txt 四个文件中的 foreach 会自动处理。
  set(TENSOR_API_OPS
      "STRMM|/trmm/arch35/strmm|strmm"
      "SDGMM|/dgmm/arch35/sdgmm|sdgmm"
      "GEMM_BATCHED|/gemm_batched/arch35/gemm_batched_|sgemm_batched,cgemm_batched"
      "SGEMM3M|/gemm3m/arch35/sgemm3m|sgemm3m"
      "SSYRK|/syrk/arch35/ssyrk|ssyrk"
      "SSYR2K|/syr2k/arch35/ssyr2k|ssyr2k"
      "SSYRKX|/syrkx/arch35/ssyrkx|ssyrkx"
      "CHERK|/herk/arch35/cherk|cherk"
      "STRSMBATCHED|/trsmbatched/arch35/strsmbatched|strsmbatched"
      "GEMM_STRIDED_BATCHED|/gemm_strided_batched/arch35/gemm_strided_batched_|gemm_strided_batched"
      "GEMM|/gemm/arch35/gemm_|gemm"
  )

  # 初始化所有 tensor_api 算子为 TRUE
  foreach(entry ${TENSOR_API_OPS})
    string(REPLACE "|" ";" _fields "${entry}")
    list(GET _fields 0 _suffix)
    set(ENABLE_BLAS_${_suffix} TRUE)
  endforeach()
  set(ENABLE_BLAS_TRSM_BLOCKED TRUE)

  if(EXISTS "${_header}")
    file(READ "${_header}" _version_content)
    if(_version_content MATCHES "#define ASC_DEVKIT_MAJOR ([0-9]+)")
      set(ASC_DEVKIT_MAJOR "${CMAKE_MATCH_1}")
    endif()
    if(_version_content MATCHES "#define ASC_DEVKIT_MINOR ([0-9]+)")
      set(ASC_DEVKIT_MINOR "${CMAKE_MATCH_1}")
    endif()
    if(ASC_DEVKIT_MAJOR GREATER_EQUAL 9 AND ASC_DEVKIT_MINOR GREATER 0)
      set(ENABLE_BLASLT_MXFP8 TRUE)
    endif()
    # arch35 的 tensor_api 算子需 devkit >= 9.1；其他架构不受限
    if("arch35" IN_LIST SOC_ARCH_DIRS AND NOT (ASC_DEVKIT_MAJOR GREATER 9 OR (ASC_DEVKIT_MAJOR EQUAL 9 AND ASC_DEVKIT_MINOR GREATER 0)))
      foreach(entry ${TENSOR_API_OPS})
        string(REPLACE "|" ";" _fields "${entry}")
        list(GET _fields 0 _suffix)
        set(ENABLE_BLAS_${_suffix} FALSE)
      endforeach()
      set(ENABLE_BLAS_TRSM_BLOCKED FALSE)
    endif()
  else()
    foreach(entry ${TENSOR_API_OPS})
      string(REPLACE "|" ";" _fields "${entry}")
      list(GET _fields 0 _suffix)
      set(ENABLE_BLAS_${_suffix} FALSE)
    endforeach()
    set(ENABLE_BLAS_TRSM_BLOCKED FALSE)
    message(WARNING "asc_devkit_version.h not found: ${_header}, tensor_api ops and TRSM_BLOCKED will be skipped")
  endif()

  set(ASC_DEVKIT_MAJOR ${ASC_DEVKIT_MAJOR} PARENT_SCOPE)
  set(ASC_DEVKIT_MINOR ${ASC_DEVKIT_MINOR} PARENT_SCOPE)
  set(ENABLE_BLASLT_MXFP8 ${ENABLE_BLASLT_MXFP8} PARENT_SCOPE)
  foreach(entry ${TENSOR_API_OPS})
    string(REPLACE "|" ";" _fields "${entry}")
    list(GET _fields 0 _suffix)
    set(ENABLE_BLAS_${_suffix} ${ENABLE_BLAS_${_suffix}} PARENT_SCOPE)
  endforeach()
  set(ENABLE_BLAS_TRSM_BLOCKED ${ENABLE_BLAS_TRSM_BLOCKED} PARENT_SCOPE)
  set(TENSOR_API_OPS ${TENSOR_API_OPS} PARENT_SCOPE)

  set(_enable_status "")
  foreach(entry ${TENSOR_API_OPS})
    string(REPLACE "|" ";" _fields "${entry}")
    list(GET _fields 0 _suffix)
    string(APPEND _enable_status ", ENABLE_BLAS_${_suffix}=${ENABLE_BLAS_${_suffix}}")
  endforeach()
  string(APPEND _enable_status ", ENABLE_BLAS_TRSM_BLOCKED=${ENABLE_BLAS_TRSM_BLOCKED}")
  message(STATUS "ASC_DEVKIT_MAJOR=${ASC_DEVKIT_MAJOR}, ASC_DEVKIT_MINOR=${ASC_DEVKIT_MINOR}, ENABLE_BLASLT_MXFP8=${ENABLE_BLASLT_MXFP8}${_enable_status}")
endfunction()
