# aclblasLt 接口文档

## 1. 模块简介

`aclblasLt` 是面向矩阵乘（GEMM）场景的轻量级高级接口，提供以下能力：

- 句柄生命周期管理（Create/Destroy）
- 矩阵布局描述（MatrixLayout）及属性配置/查询
- Matmul 操作描述（MatmulDesc）及属性配置
- 启发式算法查询（Heuristic）
- 执行矩阵乘计算（Matmul）
- 版本与属性查询（GetVersion/GetProperty）

典型计算形式：

\[
D = \alpha \cdot (A \times B) + \beta \cdot C
\]

---

## 2. 句柄与描述符类型

- `aclblasLtHandle_t`：库上下文句柄。
- `aclblasLtMatrixLayout_t`：矩阵布局描述符。
- `aclblasLtMatmulDesc_t`：矩阵乘操作描述符。
- `aclblasLtMatmulPreference_t`：算法搜索偏好描述符。
- `aclblasLtMatmulAlgo_t`：算法对象。
- `aclblasLtMatmulHeuristicResult_t`：启发式结果（含算法、工作区大小、状态等）。

---

## 3. 主要枚举

### 3.1 矩阵存储顺序 `aclblasLtOrder_t`

- `ACLBLASLT_ORDER_COL`：列主序
- `ACLBLASLT_ORDER_ROW`：行主序

### 3.2 Epilogue 类型 `aclblasLtEpilogue_t`

包含默认、ReLU、GELU、Bias 及其组合等后处理选项（详见头文件枚举定义）。

### 3.3 MatrixLayout 属性 `aclblasLtMatrixLayoutAttribute_t`

常用属性包括：

- `ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT`
- `ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`
- `ACLBLASLT_MATRIX_LAYOUT_TYPE`
- `ACLBLASLT_MATRIX_LAYOUT_ORDER`
- `ACLBLASLT_MATRIX_LAYOUT_ROWS`
- `ACLBLASLT_MATRIX_LAYOUT_COLS`
- `ACLBLASLT_MATRIX_LAYOUT_LD`

### 3.4 MatmulDesc 属性 `aclblasLtMatmulDescAttribute_t`

包括转置、epilogue、bias 指针、数据类型、scale 指针、aux 指针等属性（详见头文件枚举定义）。

### 3.5 Preference 属性 `aclblasLtMatmulPreferenceAttribute_t`

- `ACLBLASLT_MATMUL_PREF_SEARCH_MODE`
- `ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`

---

## 4. API 参考

> 说明：以下所有接口返回类型均为 `aclblasStatus_t`。

## 4.1 版本与属性查询

### 4.1.1 `aclblasLtGetVersion`

```c
aclblasStatus_t aclblasLtGetVersion(size_t* version);
```

**功能**
- 查询 aclblasLt 打包版本号。

**参数**
- `version`（输出）：版本值地址，不能为空。

**返回**
- `ACLBLAS_STATUS_SUCCESS`：成功。
- `ACLBLAS_STATUS_INVALID_VALUE`：参数非法（如 `version == NULL`）。

---

### 4.1.2 `aclblasLtGetProperty`

```c
aclblasStatus_t aclblasLtGetProperty(aclblasLtPropertyType_t type, int* value);
```

**功能**
- 查询库属性（如主版本、次版本、补丁号等）。

**参数**
- `type`（输入）：属性类型。
- `value`（输出）：属性值输出地址。

**返回**
- `ACLBLAS_STATUS_SUCCESS`：成功。
- `ACLBLAS_STATUS_INVALID_VALUE`：参数非法或属性类型不支持。

---

## 4.2 库句柄管理

### 4.2.1 `aclblasLtCreate`

```c
aclblasStatus_t aclblasLtCreate(aclblasLtHandle_t* handle);
```

**功能**
- 创建 aclblasLt 上下文句柄。

**参数**
- `handle`（输出）：返回创建的句柄。

**返回**
- `ACLBLAS_STATUS_SUCCESS`
- `ACLBLAS_STATUS_INVALID_VALUE`
- `ACLBLAS_STATUS_NOT_INITIALIZED`
- `ACLBLAS_STATUS_ALLOC_FAILED`

---

### 4.2.2 `aclblasLtDestroy`

```c
aclblasStatus_t aclblasLtDestroy(const aclblasLtHandle_t handle);
```

**功能**
- 销毁句柄并释放资源。

**参数**
- `handle`（输入）：待销毁句柄。

---

## 4.3 MatrixLayout 描述符

### 4.3.1 `aclblasLtMatrixLayoutCreate`

```c
aclblasStatus_t aclblasLtMatrixLayoutCreate(aclblasLtMatrixLayout_t* matLayout,
                                            aclDataType type,
                                            uint64_t rows,
                                            uint64_t cols,
                                            int64_t ld);
```

**功能**
- 创建矩阵布局描述符。

**参数说明**
- `type`：数据类型。
- `rows/cols`：矩阵行列。
- `ld`：leading dimension。

---

### 4.3.2 `aclblasLtMatrixLayoutDestroy`

```c
aclblasStatus_t aclblasLtMatrixLayoutDestroy(const aclblasLtMatrixLayout_t matLayout);
```

---

### 4.3.3 `aclblasLtMatrixLayoutSetAttribute`

```c
aclblasStatus_t aclblasLtMatrixLayoutSetAttribute(aclblasLtMatrixLayout_t matLayout,
                                                  aclblasLtMatrixLayoutAttribute_t attr,
                                                  const void* buf,
                                                  size_t sizeInBytes);
```

**功能**
- 设置 MatrixLayout 属性。

---

### 4.3.4 `aclblasLtMatrixLayoutGetAttribute`

```c
aclblasStatus_t aclblasLtMatrixLayoutGetAttribute(const aclblasLtMatrixLayout_t matLayout,
                                                  aclblasLtMatrixLayoutAttribute_t attr,
                                                  void* buf,
                                                  size_t sizeInBytes,
                                                  size_t* sizeWritten);
```

**功能**
- 查询 MatrixLayout 属性值。

**参数**
- `matLayout`（输入）：矩阵布局描述符。
- `attr`（输入）：要查询的属性。
- `buf`（输出）：接收属性值的缓冲区。
- `sizeInBytes`（输入）：`buf` 大小。
- `sizeWritten`（输出，可选）：实际写入字节数。

**返回**
- `ACLBLAS_STATUS_SUCCESS`
- `ACLBLAS_STATUS_INVALID_VALUE`
- `ACLBLAS_STATUS_NOT_SUPPORTED`

---

## 4.4 MatmulDesc 描述符

### 4.4.1 `aclblasLtMatmulDescCreate`

```c
aclblasStatus_t aclblasLtMatmulDescCreate(aclblasLtMatmulDesc_t* matmulDesc,
                                          aclblasComputeType_t computeType,
                                          aclDataType scaleType);
```

### 4.4.2 `aclblasLtMatmulDescDestroy`

```c
aclblasStatus_t aclblasLtMatmulDescDestroy(const aclblasLtMatmulDesc_t matmulDesc);
```

### 4.4.3 `aclblasLtMatmulDescSetAttribute`

```c
aclblasStatus_t aclblasLtMatmulDescSetAttribute(aclblasLtMatmulDesc_t matmulDesc,
                                                aclblasLtMatmulDescAttribute_t attr,
                                                const void* buf,
                                                size_t sizeInBytes);
```

---

## 4.5 MatmulPreference 描述符

### 4.5.1 `aclblasLtMatmulPreferenceCreate`

```c
aclblasStatus_t aclblasLtMatmulPreferenceCreate(aclblasLtMatmulPreference_t* pref);
```

### 4.5.2 `aclblasLtMatmulPreferenceDestroy`

```c
aclblasStatus_t aclblasLtMatmulPreferenceDestroy(const aclblasLtMatmulPreference_t pref);
```

### 4.5.3 `aclblasLtMatmulPreferenceSetAttribute`

```c
aclblasStatus_t aclblasLtMatmulPreferenceSetAttribute(aclblasLtMatmulPreference_t pref,
                                                      aclblasLtMatmulPreferenceAttribute_t attr,
                                                      const void* buf,
                                                      size_t sizeInBytes);
```

---

## 4.6 启发式算法查询

### 4.6.1 `aclblasLtMatmulAlgoGetHeuristic`

```c
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
```

**功能**
- 根据输入描述符和偏好，返回可用算法候选。

---

## 4.7 矩阵乘执行

### 4.7.1 `aclblasLtMatmul`

```c
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
```

**功能**
- 执行矩阵乘及线性组合。

**说明**
- 支持 `C == D` 的原位计算。
- `workspace` 建议满足 16B 对齐。
- 若 `algo == NULL`，实现可采用默认策略。

---

## 5. 常见调用流程（推荐）

1. `aclblasLtCreate`
2. 创建 `MatrixLayout`（A/B/C/D）并配置属性
3. 创建 `MatmulDesc` 并设置转置、epilogue 等属性
4. 创建 `MatmulPreference` 并设置工作区上限
5. `aclblasLtMatmulAlgoGetHeuristic` 查询算法
6. 调用 `aclblasLtMatmul`
7. 销毁 Preference/Desc/Layout/Handle

---

## 6. 最小示例（伪代码）

```c
aclblasLtHandle_t handle;
aclblasLtCreate(&handle);

aclblasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
aclblasLtMatrixLayoutCreate(&Adesc, ACL_FLOAT16, m, k, lda);
aclblasLtMatrixLayoutCreate(&Bdesc, ACL_FLOAT16, k, n, ldb);
aclblasLtMatrixLayoutCreate(&Cdesc, ACL_FLOAT16, m, n, ldc);
aclblasLtMatrixLayoutCreate(&Ddesc, ACL_FLOAT16, m, n, ldd);

aclblasLtMatmulDesc_t opDesc;
aclblasLtMatmulDescCreate(&opDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT);

aclblasLtMatmulPreference_t pref;
aclblasLtMatmulPreferenceCreate(&pref);
size_t workspaceCap = 32 * 1024 * 1024;
aclblasLtMatmulPreferenceSetAttribute(pref,
  ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
  &workspaceCap,
  sizeof(workspaceCap));

aclblasLtMatmulHeuristicResult_t heur[8];
int algoCount = 0;
aclblasLtMatmulAlgoGetHeuristic(handle, opDesc, Adesc, Bdesc, Cdesc, Ddesc,
                                pref, 8, heur, &algoCount);

aclblasLtMatmul(handle, opDesc,
                &alpha, A, Adesc,
                B, Bdesc,
                &beta, C, Cdesc,
                D, Ddesc,
                &heur[0].algo,
                workspace, workspaceBytes,
                stream);

aclblasLtMatmulPreferenceDestroy(pref);
aclblasLtMatmulDescDestroy(opDesc);
aclblasLtMatrixLayoutDestroy(Adesc);
aclblasLtMatrixLayoutDestroy(Bdesc);
aclblasLtMatrixLayoutDestroy(Cdesc);
aclblasLtMatrixLayoutDestroy(Ddesc);
aclblasLtDestroy(handle);
```

---

## 7. 返回码说明（通用）

- `ACLBLAS_STATUS_SUCCESS`：成功
- `ACLBLAS_STATUS_INVALID_VALUE`：参数非法
- `ACLBLAS_STATUS_NOT_INITIALIZED`：上下文未初始化
- `ACLBLAS_STATUS_NOT_SUPPORTED`：当前配置或属性不支持
- `ACLBLAS_STATUS_ALLOC_FAILED`：内存分配失败
- `ACLBLAS_STATUS_EXECUTION_FAILED`：设备执行失败（执行类接口）

---

## 8. 备注

- 接口能力与属性支持范围以当前实现版本为准。
- 文档与头文件不一致时，请以头文件声明和实际实现行为为准。
