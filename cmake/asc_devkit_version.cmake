# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 读取 asc_devkit_version.h，判断是否满足 MXFP8 所需版本：ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR > 0
function(ops_blas_detect_asc_devkit_version)
  set(_header
      "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/version/asc_devkit_version.h")
  set(ASC_DEVKIT_MAJOR 0)
  set(ASC_DEVKIT_MINOR 0)
  set(ENABLE_BLASLT_MXFP8 FALSE)

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
  else()
    message(WARNING "asc_devkit_version.h not found: ${_header}, MXFP8 will be skipped")
  endif()

  set(ASC_DEVKIT_MAJOR ${ASC_DEVKIT_MAJOR} PARENT_SCOPE)
  set(ASC_DEVKIT_MINOR ${ASC_DEVKIT_MINOR} PARENT_SCOPE)
  set(ENABLE_BLASLT_MXFP8 ${ENABLE_BLASLT_MXFP8} PARENT_SCOPE)
  message(
    STATUS
    "ASC_DEVKIT_MAJOR=${ASC_DEVKIT_MAJOR}, ASC_DEVKIT_MINOR=${ASC_DEVKIT_MINOR}, ENABLE_BLASLT_MXFP8=${ENABLE_BLASLT_MXFP8}"
  )
endfunction()
