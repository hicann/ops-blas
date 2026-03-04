/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "cann_ops_blas_common.h"
#include <acl/acl.h>

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup types_module
 *  \brief Descriptor of the matrix layout.
 */
typedef struct
{
  uint64_t data[4];
} aclblasLtMatrixLayoutOpaque_t;

typedef aclblasLtMatrixLayoutOpaque_t* aclblasLtMatrixLayout_t;

/*! \ingroup types_module
 *  \brief Descriptor of the library context.
 */
typedef void* aclblasLtHandle_t;

/*! \ingroup types_module
 *  \brief Descriptor of the matmul operation.
 */
typedef struct
{
  uint64_t data[4];
} aclblasLtMatmulDescOpaque_t;

typedef aclblasLtMatmulDescOpaque_t* aclblasLtMatmulDesc_t;

/*! \ingroup types_module
 *  \brief Descriptor of the matmul preference.
 */
typedef struct
{
  uint64_t data[5];
} aclblasLtMatmulPreferenceOpaque_t;

typedef aclblasLtMatmulPreferenceOpaque_t* aclblasLtMatmulPreference_t;

/*! \ingroup types_module
 *  \struct aclblasLtMatmulAlgo_t
 *  \brief Description of the matrix multiplication algorithm.
 */
typedef struct _aclblasLtMatmulAlgo_t {
#ifdef __cplusplus
  uint8_t data[16] = {0};
  size_t max_workspace_bytes = 0;
#else
  uint8_t data[16];
  size_t max_workspace_bytes;
#endif
} aclblasLtMatmulAlgo_t;

/*! \ingroup types_module
 *  \brief Matmul Heuristic Result.
 */
typedef struct _aclblasLtMatmulHeuristicResult_t {
  aclblasLtMatmulAlgo_t algo; /**< Algo struct */
  size_t workspaceSize;       /**< Actual size of workspace memory required. */
  aclblasStatus_t state;      /**< Result status. */
  float wavesCount;           /**< Waves count is a device utilization metric. */
  int reserved[4];            /**< Reserved. */
} aclblasLtMatmulHeuristicResult_t;

/*! \ingroup types_module
 *  \brief Matrix order.
 */
typedef enum aclblasLtOrder {
  ACLBLASLT_ORDER_COL = 0, /**< Column major */
  ACLBLASLT_ORDER_ROW = 1, /**< Row major */
} aclblasLtOrder_t;

/*! \ingroup types_module
 *  \brief  Property type.
 */
typedef enum aclblasLtPropertyType {
  ACLBLASLT_PROPERTY_MAJOR_VERSION = 0, /**<Major version number. */
  ACLBLASLT_PROPERTY_MINOR_VERSION = 1, /**<Minor version number. */
  ACLBLASLT_PROPERTY_PATCH_LEVEL = 2,   /**<Patch version number. */
} aclblasLtPropertyType_t;

/*! \ingroup types_module
 *  \brief Specifies the enumeration type to set the postprocessing options for the epilogue.
 */
typedef enum aclblasLtEpilogue {
  ACLBLASLT_EPILOGUE_DEFAULT = 1,                 /**<No special postprocessing. Scale and quantize the results if necessary.*/
  ACLBLASLT_EPILOGUE_RELU = 2,                    /**<Apply ReLU pointwise transform to the results (``x:=max(x, 0)``)*/
  ACLBLASLT_EPILOGUE_BIAS = 4,                    /**<Apply (broadcast) bias from the bias vector. The bias vector length must match the number of rows in matrix D, and it must be packed (so the stride between vector elements is one). The bias vector is broadcast to all columns and added before applying the final postprocessing.*/
  ACLBLASLT_EPILOGUE_RELU_BIAS = 6,               /**<Apply bias and then ReLU transform.*/
  ACLBLASLT_EPILOGUE_GELU = 32,                   /**<Apply GELU pointwise transform to the results (``x:=GELU(x)``).*/
  ACLBLASLT_EPILOGUE_GELU_BIAS = 36,              /**<Apply Bias and then GELU transform.*/
  ACLBLASLT_EPILOGUE_RELU_AUX = 130,              /**<Output GEMM results before applying RELU transform.*/
  ACLBLASLT_EPILOGUE_RELU_AUX_BIAS = 134,         /**<Output GEMM results after applying bias but before applying RELU transform.*/
  ACLBLASLT_EPILOGUE_DRELU = 136,
  ACLBLASLT_EPILOGUE_DRELU_BGRAD = 152,           /**<Apply gradient RELU transform and bias gradient to the results. Requires additional auxiliary input. */           /**<Apply gradient RELU transform. Requires additional auxiliary input. */
  ACLBLASLT_EPILOGUE_GELU_AUX = 160,              /**<Output GEMM results before applying GELU transform.*/
  ACLBLASLT_EPILOGUE_GELU_AUX_BIAS = 164,         /**<Output GEMM results after applying bias but before applying GELU transform.*/
  ACLBLASLT_EPILOGUE_DGELU = 192,                 /**<Apply gradient GELU transform. Requires additional auxiliary input. */
  ACLBLASLT_EPILOGUE_DGELU_BGRAD = 208,           /**<Apply gradient GELU transform and bias gradient to the results. Requires additional auxiliary input. */
  ACLBLASLT_EPILOGUE_BGRADA = 256,                /**<Apply bias gradient to A and output GEMM result. */
  ACLBLASLT_EPILOGUE_BGRADB = 512,                /**<Apply bias gradient to B and output GEMM result. */
  ACLBLASLT_EPILOGUE_SIGMOID = 1024,              /**<Apply sigmoid activation function pointwise. */
  ACLBLASLT_EPILOGUE_SWISH_EXT = 65536,           /**<Apply Swish pointwise transform to the results (``x:=Swish(x, 1)``).*/
  ACLBLASLT_EPILOGUE_SWISH_BIAS_EXT = 65540,      /**<Apply Bias and then Swish transform.*/
  ACLBLASLT_EPILOGUE_CLAMP_EXT = 131072,          /**<Apply pointwise clamp to the results (``x:=max(alpha, min(x, beta))``).*/
  ACLBLASLT_EPILOGUE_CLAMP_BIAS_EXT = 131076,     /**<Apply Bias and then clamp.*/
  ACLBLASLT_EPILOGUE_CLAMP_AUX_EXT = 131200,      /**<Output GEMM results before applying clamp transform.*/
  ACLBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT = 131204, /**<Output GEMM results after applying bias but before applying clamp transform.*/
} aclblasLtEpilogue_t;

/*! \ingroup types_module
 *  \brief Matrix layout attributes.
 */
typedef enum aclblasLtMatrixLayoutAttribute {
  ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 0,         /**<Number of batches of this matrix. Default value is 1. Data type: ``int32_t``. */
  ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 1, /**<Stride (in elements) to the next matrix for the strided batch operation. Default value is 0. Data type: ``int64_t``. */
  /** Data type. See ``aclDataType``.
   *
   * ``uint32_t``
   */
  ACLBLASLT_MATRIX_LAYOUT_TYPE = 2,

  /** Memory order of the data. See ``aclblasLtOrder_t``.
   *
   * ``int32_t``, default: ``ACLBLASLT_ORDER_COL``.
   */
  ACLBLASLT_MATRIX_LAYOUT_ORDER = 3,

  /** Number of rows.
   *
   * Typically only values that can be expressed as ``int32_t`` are supported.
   *
   * ``uint64_t``
   */
  ACLBLASLT_MATRIX_LAYOUT_ROWS = 4,

  /** Number of columns.
   *
   * Typically only values that can be expressed as ``int32_t`` are supported.
   *
   * ``uint64_t``
   */
  ACLBLASLT_MATRIX_LAYOUT_COLS = 5,

  /** Matrix leading dimension.
   *
   * For ``ACLBLASLT_ORDER_COL``, this is the stride (in elements) of the matrix column. For more details and documentation for
   * other memory orders, see the documentation for ``a c lblasLtOrder_t`` values.
   *
   * Currently only non-negative values are supported. The value must be large enough so that matrix memory locations are not
   * overlapping (that is, greater or equal to ``ACLBLASLT_MATRIX_LAYOUT_ROWS`` in the case of ``ACLBLASLT_ORDER_COL``).
   *
   * ``int64_t``
   */
  ACLBLASLT_MATRIX_LAYOUT_LD = 6,
} aclblasLtMatrixLayoutAttribute_t;

/*! \ingroup types_module
 *  \brief Matmul operation descriptor attributes.
 */
typedef enum aclblasLtMatmulDescAttribute {
  ACLBLASLT_MATMUL_DESC_TRANSA = 0,                     /**<Specifies the type of transformation operation that should be performed on matrix A. Default value is ``ACLBLAS_OP_N`` (for example, non-transpose operation). See ``aclblasOperation_t``. Data type: ``int32_t``. */
  ACLBLASLT_MATMUL_DESC_TRANSB = 1,                     /**<Specifies the type of transformation operation that should be performed on matrix B. Default value is ``ACLBLAS_OP_N`` (for example, non-transpose operation). See ``aclblasOperation_t``. Data type: ``int32_t``. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE = 2,                   /**<Epilogue function. See ``aclblasLtEpilogue_t``. Default value is ``ACLBLASLT_EPILOGUE_DEFAULT``. Data type: ``uint32_t``. */
  ACLBLASLT_MATMUL_DESC_BIAS_POINTER = 3,               /**<Bias or bias gradient vector pointer in the device memory. Data type: ``void*`` / ``const void*``. */
  ACLBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 4,             /**<Type of the bias vector in the device memory. Can be set the same as the D matrix type or Scale type. Bias case: see ``ACLBLASLT_EPILOGUE_BIAS``. Data type: ``int32_t`` based on ``aclDataType``. */
  ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER = 5,            /**<Device pointer to the scale factor value that converts data in matrix A to the compute data type range. The scaling factor must have the same type as the compute type. If not specified, or set to NULL, the scaling factor is assumed to be ``1``. If set for an unsupported matrix data, scale, and compute type combination, calling aclblasLtMatmul() will return ``ACLBLAS_INVALID_VALUE``. Default value: NULL. Data type: ``void*`` ``/const void*``. */
  ACLBLASLT_MATMUL_DESC_B_SCALE_POINTER = 6,            /**<Equivalent to ``ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER`` for matrix B. Default value: NULL. Data type: ``void*`` / ``const void*``. */
  ACLBLASLT_MATMUL_DESC_C_SCALE_POINTER = 7,            /**<Equivalent to ``ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER`` for matrix C. Default value: NULL. Data type: ``void*`` / ``const void*``. */
  ACLBLASLT_MATMUL_DESC_D_SCALE_POINTER = 8,            /**<Equivalent to ``ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER`` for matrix D. Default value: NULL. Data type: ``void*`` / ``const void*``. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = 9, /**<Equivalent to ``ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER`` for matrix AUX. Default value: NULL. Data type: ``void*`` / ``const void*``. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 10,      /**<Epilogue auxiliary buffer pointer in the device memory. Data type: ``void*`` / ``const void*``. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 11,           /**<The leading dimension of the epilogue auxiliary buffer pointer in the device memory. Data type: ``int64_t``. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 12, /**<The batch stride of the epilogue auxiliary buffer pointer in the device memory. Data type: ``int64_t``. */
  ACLBLASLT_MATMUL_DESC_POINTER_MODE = 13,              /**<Specifies that alpha and beta are passed by reference, whether they are scalars on the host or on the device, or device vectors. Default value is: ``ACLBLASLT_POINTER_MODE_HOST`` (on the host). Data type: ``int32_t`` based on ``aclblasLtPointerMode_t``. */
  ACLBLASLT_MATMUL_DESC_AMAX_D_POINTER = 14,           /**<Device pointer to the memory location that on completion will be set to the maximum of the absolute values in the output matrix. Data type: ``void*`` / ``const void*``. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = 22,    /**<Type of the auxiliary vector in the device memory. Default value is: ``ACLBLASLT_DATATYPE_INVALID`` (using D matrix type). Data type: ``int32_t`` based on ``aclDataType``. */
  ACLBLASLT_MATMUL_DESC_A_SCALE_MODE = 31,                   /**<Scaling mode that defines how the matrix scaling factor for matrix A is interpreted. See ``aclblasLtMatmulMatrixScale_t``. */
  ACLBLASLT_MATMUL_DESC_B_SCALE_MODE = 32,                   /**<Scaling mode that defines how the matrix scaling factor for matrix B is interpreted. See ``aclblasLtMatmulMatrixScale_t``. */
  ACLBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT = 100,     /**<Compute input A types. Defines the data type used for the input A of a matrix multiply. */
  ACLBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT,           /**<Compute input B types. Defines the data type used for the input B of a matrix multiply. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT,              /**<First extra argument for the activation function. Data type: ``float``. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT,              /**<Second extra argument for the activation function. Data type: ``float``. */
  ACLBLASLT_MATMUL_DESC_MAX,
} aclblasLtMatmulDescAttribute_t;

/*! \ingroup types_module
 *  \brief Matmul preference attributes.
 */
typedef enum aclblasLtMatmulPreferenceAttribute {
  ACLBLASLT_MATMUL_PREF_SEARCH_MODE = 0,          /**<Search mode. Data type: ``uint32_t``. */
  ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,  /**<Maximum allowed workspace memory. Default is 0 (no workspace memory allowed). Data type: ``uint64_t``. */
  ACLBLASLT_MATMUL_PREF_MAX = 2
} aclblasLtMatmulPreferenceAttribute_t;

/*! \ingroup library_module
 *  \brief Query aclBLASLt packed version number.
 */
 aclblasStatus_t aclblasLtGetVersion(size_t* version);

 /*! \ingroup library_module
  *  \brief Query aclBLASLt property value.
  */
 aclblasStatus_t aclblasLtGetProperty(aclblasLtPropertyType_t type, int* value);

// Library management
/*! \ingroup library_module
 *  \brief Create a aclBLASLt handle.
 *
 *  \details
 *  This function initializes the aclBLASLt library and creates a handle to an
 *  opaque structure holding the aclBLASLt library context. It allocates light
 *  hardware resources on the host and device and must be called prior to making
 *  any other aclBLASLt library calls. The aclBLASLt library context is tied to
 *  the current CANN device. To use the library on multiple devices, one
 *  aclBLASLt handle should be created for each device.
 *
 *  @param[out]
 *  handle Pointer to the allocated aclBLASLt handle for the created aclBLASLt
 *  context.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS The allocation completed successfully.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE \p handle == NULL.
 */
aclblasStatus_t aclblasLtCreate(aclblasLtHandle_t* handle);

/*! \ingroup library_module
 *  \brief Destroy a aclBLASLt handle.
 *
 *  \details
 *  This function releases hardware resources used by the aclBLASLt library.
 *  It is usually the last call with a particular handle to the
 *  aclBLASLt library. Because aclblasLtCreate() allocates some internal
 *  resources and the release of those resources by calling aclblasLtDestroy()
 *  implicitly calls device synchronization, it is recommended to minimize
 *  the number of aclblasLtCreate() / aclblasLtDestroy() occurrences.
 *
 *  @param[in]
 *  handle Pointer to the aclBLASLt handle to be destroyed.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS The aclBLASLt context was successfully
 *  destroyed. \retval ACLBLAS_STATUS_NOT_INITIALIZED The aclBLASLt library was
 *  not initialized. \retval ACLBLAS_STATUS_INVALID_VALUE \p handle == NULL.
 */
aclblasStatus_t aclblasLtDestroy(const aclblasLtHandle_t handle);

// Matrix layout descriptor
/*! \ingroup library_module
 *  \brief Create a matrix layout descriptor.
 *
 *  \details
 *  This function creates a matrix layout descriptor by allocating the memory
 *  needed to hold its opaque structure.
 *
 *  @param[out]
 *  matLayout Pointer to the structure holding the matrix layout descriptor
 *  created by this function. See \ref aclblasLtMatrixLayout_t.
 *  @param[in]
 *  type Enumerant that specifies the data precision for the matrix layout
 *  descriptor created by this function. See aclDataType.
 *  @param[in]
 *  rows Number of rows of the matrix.
 *  @param[in]
 *  cols Number of columns of the matrix.
 *  @param[in]
 *  ld The leading dimension of the matrix. In column major layout, this is the
 *  number of elements to jump to reach the next column. Therefore, ld >= m (number of
 *  rows).
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the descriptor was created successfully.
 *  \retval ACLBLAS_STATUS_ALLOC_FAILED If the memory could not be allocated.
 */
aclblasStatus_t aclblasLtMatrixLayoutCreate(aclblasLtMatrixLayout_t* matLayout,
                                           aclDataType type,
                                           uint64_t rows,
                                           uint64_t cols,
                                           int64_t ld);

/*! \ingroup library_module
 *  \brief Destroy a matrix layout descriptor.
 *
 *  \details
 *  This function destroys a previously created matrix layout descriptor object.
 *
 *  @param[in]
 *  matLayout Pointer to the structure holding the matrix layout descriptor to
 *  be destroyed by this function. See \ref aclblasLtMatrixLayout_t.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the operation was successful.
 */
aclblasStatus_t aclblasLtMatrixLayoutDestroy(const aclblasLtMatrixLayout_t matLayout);

/*! \ingroup library_module
 *  \brief Set an attribute for a matrix descriptor.
 *
 *  \details
 *  This function sets the value of the specified attribute belonging to a
 *  previously created matrix descriptor.
 *
 *  @param[in]
 *  matLayout Pointer to the previously created structure holding the matrix
 *  descriptor queried by this function. See \ref aclblasLtMatrixLayout_t.
 *  @param[in]
 *  attr The attribute that will be set by this function. See \ref
 *  aclblasLtMatrixLayoutAttribute_t.
 *  @param[in]
 *  buf The value to which the specified attribute should be set.
 *  @param[in]
 *  sizeInBytes Size of the buf buffer (in bytes) for verification.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the attribute was set successfully.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p buf is NULL or \p sizeInBytes
 *  doesn't match the size of the internal storage for the selected attribute.
 */
aclblasStatus_t aclblasLtMatrixLayoutSetAttribute(aclblasLtMatrixLayout_t matLayout,
                                                 aclblasLtMatrixLayoutAttribute_t attr,
                                                 const void* buf,
                                                 size_t sizeInBytes);

/*! \ingroup library_module
 *  \brief Get an attribute for a matrix descriptor.
 */
aclblasStatus_t aclblasLtMatrixLayoutGetAttribute(const aclblasLtMatrixLayout_t matLayout,
    aclblasLtMatrixLayoutAttribute_t attr,
    void* buf,
    size_t sizeInBytes,
    size_t* sizeWritten);

// Matmul operation descriptor
/*! \ingroup library_module
 *  \brief Create a matrix multiply descriptor.
 *
 *  \details
 *  This function creates a matrix multiply descriptor by allocating the memory
 *  needed to hold its opaque structure.
 *
 *  @param[out]
 *  matmulDesc Pointer to the structure holding the matrix multiply descriptor
 *  created by this function. See \ref aclblasLtMatmulDesc_t.
 *  @param[in]
 *  computeType Enumerant that specifies the data precision for the matrix
 *  multiply descriptor this function creates. See aclblasComputeType_t.
 *  @param[in]
 *  scaleType Enumerant that specifies the data precision for the matrix
 *  transform descriptor this function creates. See aclDataType.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the descriptor was created successfully.
 *  \retval ACLBLAS_STATUS_ALLOC_FAILED If the memory could not be allocated.
 */
aclblasStatus_t aclblasLtMatmulDescCreate(aclblasLtMatmulDesc_t* matmulDesc,
                                         aclblasComputeType_t computeType,
                                         aclDataType scaleType);

/*! \ingroup library_module
 *  \brief Destroy a matrix multiply descriptor.
 *
 *  \details
 *  This function destroys a previously created matrix multiply descriptor
 *  object.
 *
 *  @param[in]
 *  matmulDesc Pointer to the structure holding the matrix multiply descriptor
 *  to be destroyed by this function. See \ref aclblasLtMatmulDesc_t.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If operation was successful.
 */
aclblasStatus_t aclblasLtMatmulDescDestroy(const aclblasLtMatmulDesc_t matmulDesc);

/*! \ingroup library_module
 *  \brief Set attribute to a matrix multiply descriptor.
 *
 *  \details
 *  This function sets the value of the specified attribute belonging to a
 *  previously created matrix multiply descriptor.
 *
 *  @param[in]
 *  matmulDesc Pointer to the previously created structure holding the matrix
 *  multiply descriptor queried by this function. See \ref aclblasLtMatmulDesc_t.
 *  @param[in]
 *  attr The attribute that will be set by this function. See \ref
 *  aclblasLtMatmulDescAttribute_t.
 *  @param[in]
 *  buf The value to which the specified attribute should be set.
 *  @param[in]
 *  sizeInBytes Size of the buf buffer (in bytes) for verification.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the attribute was set successfully.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p buf is NULL or \p sizeInBytes
 *  doesn't match the size of the internal storage for the selected attribute.
 */
aclblasStatus_t aclblasLtMatmulDescSetAttribute(aclblasLtMatmulDesc_t matmulDesc,
                                               aclblasLtMatmulDescAttribute_t attr,
                                               const void* buf,
                                               size_t sizeInBytes);

// Preference
/*! \ingroup library_module
 *  \brief Create a preference descriptor.
 *
 *  \details
 *  This function creates a matrix multiply heuristic search preferences
 *  descriptor by allocating the memory needed to hold its opaque structure.
 *
 *  @param[out]
 *  pref Pointer to the structure holding the matrix multiply preferences
 *  descriptor created by this function. see \ref aclblasLtMatmulPreference_t.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the descriptor was created
 *  successfully. \retval ACLBLAS_STATUS_ALLOC_FAILED If memory could not be
 *  allocated.
 */
aclblasStatus_t aclblasLtMatmulPreferenceCreate(aclblasLtMatmulPreference_t* pref);

/*! \ingroup library_module
 *  \brief Destroy a preference descriptor.
 *
 *  \details
 *  This function destroys a previously created matrix multiply preferences
 *  descriptor object.
 *
 *  @param[in]
 *  pref Pointer to the structure holding the matrix multiply preferences
 *  descriptor to be destroyed by this function. See \ref
 *  aclblasLtMatmulPreference_t.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If operation was successful.
 */
aclblasStatus_t aclblasLtMatmulPreferenceDestroy(const aclblasLtMatmulPreference_t pref);

/*! \ingroup library_module
 *  \brief Set attribute in a preference descriptor.
 *
 *  \details
 *  This function sets the value of the specified attribute belonging to a
 *  previously created matrix multiply preferences descriptor.
 *
 *  @param[in]
 *  pref Pointer to the previously created structure holding the matrix
 *  multiply preferences descriptor queried by this function. See \ref
 *  aclblasLtMatmulPreference_t.
 *  @param[in]
 *  attr The attribute that will be set by this function. See \ref
 *  aclblasLtMatmulPreferenceAttribute_t.
 *  @param[in]
 *  buf The value to which the specified attribute should be set.
 *  @param[in]
 *  sizeInBytes Size of the \p buf buffer (in bytes) for verification.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the attribute was set successfully.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p buf is NULL or \p sizeInBytes
 *  doesn't match the size of the internal storage for the selected attribute.
 */
aclblasStatus_t aclblasLtMatmulPreferenceSetAttribute(aclblasLtMatmulPreference_t pref,
                                                     aclblasLtMatmulPreferenceAttribute_t attr,
                                                     const void* buf,
                                                     size_t sizeInBytes);

// Heuristic + matmul
/*! \ingroup library_module
 *  \brief Retrieve the possible algorithms.
 *
 *  \details
 *  This function retrieves the possible algorithms for the matrix multiply
 *  operation aclblasLtMatmul() with the given input matrices A, B, and
 *  C, and the output matrix D. The output is placed in ``heuristicResultsArray``
 *  in order of increasing estimated compute time.
 *
 *  @param[in]
 *  handle Pointer to the allocated aclBLASLt handle for the
 *  aclBLASLt context. See \ref aclblasLtHandle_t.
 *  @param[in]
 *  matmulDesc Handle to a previously created matrix multiplication
 *  descriptor of type \ref aclblasLtMatmulDesc_t.
 *  @param[in]
 *  Adesc,Bdesc,Cdesc,Ddesc Handles to the previously created matrix layout
 *  descriptors of the type \ref aclblasLtMatrixLayout_t.
 *  @param[in]
 *  pref Pointer to the structure holding the heuristic
 *  search preferences descriptor. See \ref aclblasLtMatmulPreference_t.
 *  @param[in]
 *  requestedAlgoCount Size of the \p heuristicResultsArray (in elements).
 *  This is the requested maximum number of algorithms to return.
 *  @param[out]
 *  heuristicResultsArray[] Array containing the algorithm heuristics and
 *  associated runtime characteristics returned by this function, in order
 *  of increasing estimated compute time.
 *  @param[out]
 *  returnAlgoCount Number of algorithms returned by this function. This
 *  is the number of \p heuristicResultsArray elements written.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If query was successful. Inspect
 *  ``heuristicResultsArray[0 to (returnAlgoCount -1)].state`` for the status of the
 *  results. \retval ACLBLAS_STATUS_NOT_SUPPORTED If no heuristic function is
 *  available for current configuration. \retval ACLBLAS_STATUS_INVALID_VALUE If
 *  \p requestedAlgoCount is less than or equal to zero.
 */
aclblasStatus_t aclblasLtMatmulAlgoGetHeuristic(aclblasLtHandle_t handle,
                                               aclblasLtMatmulDesc_t matmulDesc,
                                               aclblasLtMatrixLayout_t Adesc,
                                               aclblasLtMatrixLayout_t Bdesc,
                                               aclblasLtMatrixLayout_t Cdesc,
                                               aclblasLtMatrixLayout_t Ddesc,
                                               aclblasLtMatmulPreference_t pref,
                                               int requestedAlgoCount,
                                               aclblasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                               int* returnAlgoCount);

/*! \ingroup library_module
 *  \brief Matrix-matrix multiplication.
 *
 *  \details
 *  This function computes the matrix multiplication of matrices A and B to
 *  produce the output matrix D, according to the following operation: \p D = \p
 *  alpha*( \p A *\p B) + \p beta*( \p C ), where \p A, \p B, and \p C are input
 *  matrices, and \p alpha and \p beta are input scalars. Note: This function
 *  supports both in-place matrix multiplication (``C == D`` and ``Cdesc == Ddesc``) and
 *  out-of-place matrix multiplication (``C != D``).
 *
 *  @param[in]
 *  handle Pointer to the allocated aclBLASLt handle for the
 *  aclBLASLt context. See \ref aclblasLtHandle_t.
 *  @param[in]
 *  matmulDesc Handle to a previously created matrix multiplication
 *  descriptor of type \ref aclblasLtMatmulDesc_t.
 *  @param[in]
 *  alpha,beta Pointers to the scalars used in the multiplication.
 *  @param[in]
 *  Adesc,Bdesc,Cdesc,Ddesc Handles to the previously created matrix layout
 *  descriptors of the type \ref aclblasLtMatrixLayout_t.
 *  @param[in]
 *  A,B,C Pointers to the memory associated with the
 *  corresponding descriptors \p Adesc, \p Bdesc, and \p Cdesc.
 *  @param[out]
 *  D Pointer to the memory associated with the
 *  descriptor \p Ddesc.
 *  @param[in]
 *  algo Handle for matrix multiplication algorithm to be
 *  used. See \ref aclblasLtMatmulAlgo_t. When NULL, an implicit heuristics query
 *  with default search preferences will be performed to determine the actual
 *  algorithm to use.
 *  @param[in]
 *  workspace Pointer to the workspace buffer allocated in the GPU
 *  memory. Pointer must be 16B aligned.
 *  @param[in]
 *  workspaceSizeInBytes Size of the workspace.
 *  @param[in]
 *  stream The stream where all device work is submitted.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the operation completed
 *  successfully. \retval ACLBLAS_STATUS_EXECUTION_FAILED If device reported an
 *  execution error. \retval ACLBLAS_STATUS_ARCH_MISMATCH If
 *  the configured operation cannot be run using the selected device. \retval
 *  ACLBLAS_STATUS_NOT_SUPPORTED If the current implementation on the
 *  selected device doesn't support the configured operation. \retval
 *  ACLBLAS_STATUS_INVALID_VALUE If the parameters are unexpectedly NULL, in
 *  conflict, or in an impossible configuration. \retval ACLBLAS_STATUS_NOT_INITIALIZED
 *  If the aclBLASLt handle has not been initialized.
 */
aclblasStatus_t aclblasLtMatmul(aclblasLtHandle_t handle,
                               aclblasLtMatmulDesc_t matmulDesc,
                               const void* alpha,
                               const void* A,
                               aclblasLtMatrixLayout_t Adesc,
                               const void* B,
                               aclblasLtMatrixLayout_t Bdesc,
                               const void* beta,
                               const void* C,
                               aclblasLtMatrixLayout_t Cdesc,
                               void* D,
                               aclblasLtMatrixLayout_t Ddesc,
                               const aclblasLtMatmulAlgo_t* algo,
                               void* workspace,
                               size_t workspaceSizeInBytes,
                               aclrtStream stream);

#ifdef __cplusplus
}
#endif