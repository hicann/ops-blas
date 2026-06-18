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

# CANN 9.1.0 SDK common_types.h 已定义 fp4x2_e2m1_t / fp4x2_e1m2_t 等类型，
# 与 ops-tensor macro_impl.h 的 #else 分支 uint8_t 回退定义冲突。
# 直接删除回退定义，依赖 CANN SDK 提供的类型定义。
function(ops_blas_patch_ops_tensor_fp4_types optensor_root)
  if(NOT optensor_root)
    message(FATAL_ERROR "ops_blas_patch_ops_tensor_fp4_types: optensor_root is required")
  endif()

  set(_target_file "${optensor_root}/include/tensor_api/impl/tensor_api/utils/macro_impl.h")
  if(NOT EXISTS "${_target_file}")
    message(WARNING "ops-tensor patch skipped: ${_target_file} not found")
    return()
  endif()

  file(READ "${_target_file}" _content)
  
  # 检查是否已经打过补丁（回退定义已被删除）
  if(NOT _content MATCHES "using fp4x2_e2m1_t = uint8_t;")
    message(STATUS "ops-tensor patch: macro_impl.h already patched for FP4 types, skip")
    return()
  endif()

  set(_old_block
"#else
    using fp4x2_e2m1_t = uint8_t;
    using fp4x2_e1m2_t = uint8_t;
    using fp8_e5m2_t = uint8_t;
    using fp8_e4m3fn_t = uint8_t;
    using fp8_e8m0_t = uint8_t;
#endif")

  set(_new_block
"#else
    // CANN 9.1.0 SDK common_types.h already defines these types.
    // Fallback uint8_t aliases removed to avoid redefinition conflict.
#endif")

  string(REPLACE "${_old_block}" "${_new_block}" _new_content "${_content}")
  file(WRITE "${_target_file}" "${_new_content}")
  message(STATUS "ops-tensor patch: removed FP4 fallback types to avoid CANN 9.1.0 conflict")

  # Patch kernel_qbmm_mx.h: qualify fp8_e8m0_t with :: to resolve ambiguity
  # between CANN SDK global scope (float8_e8m0_t) and AscendC namespace (uint8_t)
  set(_qbmm_file "${optensor_root}/include/blaze/kernel/kernel_qbmm_mx.h")
  if(EXISTS "${_qbmm_file}")
    file(READ "${_qbmm_file}" _qbmm_content)
    if(_qbmm_content MATCHES "fp8_e8m0_t" AND NOT _qbmm_content MATCHES "::fp8_e8m0_t")
      string(REPLACE "fp8_e8m0_t" "::fp8_e8m0_t" _qbmm_new "${_qbmm_content}")
      file(WRITE "${_qbmm_file}" "${_qbmm_new}")
      message(STATUS "ops-tensor patch: qualified fp8_e8m0_t with :: in kernel_qbmm_mx.h")
    endif()
  endif()
endfunction()
