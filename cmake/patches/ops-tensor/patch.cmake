# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# Captured at include() time; CMAKE_CURRENT_LIST_DIR inside functions refers to the caller.
set(_OPS_BLAS_OPTENSOR_PATCH_DIR "${CMAKE_CURRENT_LIST_DIR}")

# CANN 公开 API 为 GetTaskRatio(), GetTaskRation() 仅在 impl 中作为兼容别名且依赖 __NPU_ARCH__。
function(ops_blas_patch_ops_tensor_get_task_ratio optensor_root)
  if(NOT optensor_root)
    message(FATAL_ERROR "ops_blas_patch_ops_tensor_get_task_ratio: optensor_root is required")
  endif()

  set(_patched_count 0)
  foreach(_subdir IN ITEMS blaze tensor_api)
    set(_dir "${optensor_root}/include/${_subdir}")
    if(NOT EXISTS "${_dir}")
      continue()
    endif()
    file(GLOB_RECURSE _files
      LIST_DIRECTORIES false
      "${_dir}/*.h" "${_dir}/*.hpp" "${_dir}/*.cpp")
    foreach(_file IN LISTS _files)
      file(READ "${_file}" _content)
      if(_content MATCHES "GetTaskRation\\(")
        string(REPLACE "GetTaskRation(" "GetTaskRatio(" _new_content "${_content}")
        file(WRITE "${_file}" "${_new_content}")
        file(RELATIVE_PATH _rel "${optensor_root}" "${_file}")
        message(STATUS "ops-tensor patch: GetTaskRation -> GetTaskRatio in ${_rel}")
        math(EXPR _patched_count "${_patched_count} + 1")
      endif()
    endforeach()
  endforeach()
  if(_patched_count GREATER 0)
    message(STATUS "ops-tensor patch: updated ${_patched_count} header(s) for GetTaskRatio API")
  endif()
endfunction()

# bisheng AICore 不允许将 MakeMemPtr/MakeTensor 推导类型声明为类成员变量（block_mmad_mx.h）。
function(ops_blas_patch_ops_tensor_block_mmad_mx optensor_root)
  if(NOT optensor_root)
    message(FATAL_ERROR "ops_blas_patch_ops_tensor_block_mmad_mx: optensor_root is required")
  endif()

  set(_patch_file "${_OPS_BLAS_OPTENSOR_PATCH_DIR}/block_mmad_mx.h")
  set(_target_file "${optensor_root}/include/blaze/block/block_mmad_mx.h")
  if(NOT EXISTS "${_patch_file}")
    message(FATAL_ERROR "ops-tensor patch not found: ${_patch_file}")
  endif()
  if(NOT EXISTS "${_target_file}")
    message(WARNING "ops-tensor patch skipped: ${_target_file} not found")
    return()
  endif()
  file(READ "${_target_file}" _current_content)
  if(_current_content MATCHES "using TensorL1 = decltype\\(AscendC::Te::MakeTensor")
    file(COPY "${_patch_file}" DESTINATION "${optensor_root}/include/blaze/block/" FILE_PERMISSIONS
                                                                                      OWNER_READ
                                                                                      OWNER_WRITE
                                                                                      GROUP_READ
                                                                                      WORLD_READ)
    message(STATUS "ops-tensor patch: replace blaze/block/block_mmad_mx.h (local tensor members -> stack tensors)")
  else()
    message(STATUS "ops-tensor patch: blaze/block/block_mmad_mx.h already patched, skip")
  endif()
endfunction()
