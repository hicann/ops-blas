# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(NOT DEFINED CANN_3RD_LIB_PATH)
  set(CANN_3RD_LIB_PATH "${CMAKE_BINARY_DIR}/third_party")
endif()

set(OPTENSOR_TAG_ID a76e9c60f8c70b1cc906714e0ef41150a857d3ae)

if(EXISTS "${CANN_3RD_LIB_PATH}/ops-tensor")
  get_filename_component(OPTENSOR_SOURCE_PATH ${CANN_3RD_LIB_PATH}/ops-tensor REALPATH)
  message(STATUS "Find ops-tensor source dir: ${OPTENSOR_SOURCE_PATH}")
  execute_process(
    COMMAND git checkout ${OPTENSOR_TAG_ID}
    WORKING_DIRECTORY ${OPTENSOR_SOURCE_PATH}
    RESULT_VARIABLE EXEC_RESULT
    OUTPUT_VARIABLE EXEC_INFO
    ERROR_VARIABLE EXEC_ERROR
  )
  if(${EXEC_RESULT})
    message(FATAL_ERROR "Git checkout failed! error: ${EXEC_ERROR}")
  endif()
else()
  include(FetchContent)

  FetchContent_Declare(
    ops-tensor
    GIT_REPOSITORY https://gitcode.com/wangzitao_leo/ops-tensor.git
    GIT_TAG ${OPTENSOR_TAG_ID}
    GIT_PROGRESS TRUE
    SOURCE_DIR ${CANN_3RD_LIB_PATH}/ops-tensor)

  FetchContent_Populate(ops-tensor)

  set(OPTENSOR_SOURCE_PATH ${CANN_3RD_LIB_PATH}/ops-tensor)
endif()

set(OPTENSOR_INCLUDE_DIR "${OPTENSOR_SOURCE_PATH}/include")

if(NOT EXISTS "${OPTENSOR_INCLUDE_DIR}/tensor_api" AND NOT EXISTS "${OPTENSOR_INCLUDE_DIR}/blaze")
  message(
    FATAL_ERROR
      "ops-tensor headers not found: expected tensor_api/ or blaze/ under ${OPTENSOR_INCLUDE_DIR}. "
      "Set sibling clone at ../ops-tensor or ensure FetchContent succeeded.")
endif()
