# aclblasGemmGroupedBatchedEx

## 接口说明

`aclblasGemmGroupedBatchedEx` 在一次调用中执行多组列主序 GEMM。每个组共享矩阵形状、
转置方式、leading dimension、缩放系数和数据类型；同组内可以包含多个独立矩阵实例。

```c
aclblasStatus_t aclblasGemmGroupedBatchedEx(
    aclblasHandle_t handle,
    const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[],
    const void* alphaArray,
    const void* const Aarray[], aclDataType Atype, const int ldaArray[],
    const void* const Barray[], aclDataType Btype, const int ldbArray[],
    const void* betaArray,
    void* const Carray[], aclDataType Ctype, const int ldcArray[],
    int groupCount, const int groupSize[], aclblasComputeType_t computeType);
```

对于组 `g` 中的每个实例，接口计算：

```text
C[index] = alpha[g] * op(A[index]) * op(B[index]) + beta[g] * C[index]
```

其中 `index` 按组顺序连续编号，实例总数为 `sum(groupSize[g])`。

## 参数

| 参数 | 位置 | 说明 |
| --- | --- | --- |
| `handle` | Host | BLAS handle；其中的 stream 用于异步下发设备任务。不可为空。 |
| `transaArray` | Host | 长度为 `groupCount`。每组 A 的操作，可为 `ACLBLAS_OP_N`、`ACLBLAS_OP_T` 或 `ACLBLAS_OP_C`；实数类型下 `C` 等同于 `T`。 |
| `transbArray` | Host | 长度为 `groupCount`。每组 B 的操作，取值同 `transaArray`。 |
| `mArray` | Host | 长度为 `groupCount`。每组输出矩阵 C 的行数，必须大于或等于 0。 |
| `nArray` | Host | 长度为 `groupCount`。每组输出矩阵 C 的列数，必须大于或等于 0。 |
| `kArray` | Host | 长度为 `groupCount`。每组乘法的规约维度，必须大于或等于 0。 |
| `alphaArray` | Host | 长度为 `groupCount`。`computeType` 为 `ACLBLAS_COMPUTE_16F` 时元素按 FP16 存储；为 `ACLBLAS_COMPUTE_32F` 时元素按 FP32 存储。 |
| `Aarray` | Device | 长度为 `sum(groupSize)` 的设备指针数组；每个元素指向一个列主序 A 矩阵。 |
| `Atype` | Host | 所有 A 矩阵的元素类型。支持范围见下表。 |
| `ldaArray` | Host | 长度为 `groupCount`。A 的 leading dimension；非转置时至少为 `max(1, m[g])`，转置时至少为 `max(1, k[g])`。 |
| `Barray` | Device | 长度为 `sum(groupSize)` 的设备指针数组；每个元素指向一个列主序 B 矩阵。 |
| `Btype` | Host | 所有 B 矩阵的元素类型。FP8 输入允许 E4M3 和 E5M2 混合使用。 |
| `ldbArray` | Host | 长度为 `groupCount`。B 的 leading dimension；非转置时至少为 `max(1, k[g])`，转置时至少为 `max(1, n[g])`。 |
| `betaArray` | Host | 长度为 `groupCount`。元素存储类型与 `alphaArray` 相同。 |
| `Carray` | Device | 长度为 `sum(groupSize)` 的设备指针数组；每个元素指向一个列主序输入/输出 C 矩阵。 |
| `Ctype` | Host | 所有 C 矩阵的元素类型。支持范围见下表。 |
| `ldcArray` | Host | 长度为 `groupCount`。C 的 leading dimension，至少为 `max(1, m[g])`。 |
| `groupCount` | Host | 分组数量，必须大于或等于 0；为 0 时接口直接返回成功。 |
| `groupSize` | Host | 长度为 `groupCount`。每组包含的矩阵实例数，每个值必须大于或等于 0。 |
| `computeType` | Host | 缩放系数和计算精度类型，支持 `ACLBLAS_COMPUTE_16F` 或 `ACLBLAS_COMPUTE_32F`，具体限制见下表。 |

`Aarray`、`Barray`、`Carray` 本身以及它们指向的矩阵均位于 Device；其余数组位于 Host。
当实例总数大于 0 时，三个设备指针数组均不可为空。

## 支持的数据类型

| Atype | Btype | Ctype | computeType |
| --- | --- | --- | --- |
| `ACL_FLOAT16` | `ACL_FLOAT16` | `ACL_FLOAT16` | `ACLBLAS_COMPUTE_16F`、`ACLBLAS_COMPUTE_32F` |
| `ACL_BF16` | `ACL_BF16` | `ACL_BF16` | `ACLBLAS_COMPUTE_32F` |
| `ACL_FLOAT8_E4M3FN` | `ACL_FLOAT8_E4M3FN` | `ACL_FLOAT16` | `ACLBLAS_COMPUTE_32F` |
| `ACL_FLOAT8_E5M2` | `ACL_FLOAT8_E5M2` | `ACL_FLOAT16` | `ACLBLAS_COMPUTE_32F` |
| `ACL_FLOAT8_E4M3FN` | `ACL_FLOAT8_E5M2` | `ACL_FLOAT16` | `ACLBLAS_COMPUTE_32F` |
| `ACL_FLOAT8_E5M2` | `ACL_FLOAT8_E4M3FN` | `ACL_FLOAT16` | `ACLBLAS_COMPUTE_32F` |

当前目标平台为 Ascend 950（DAV_3510 / arch35）。

## 返回值

- `ACLBLAS_STATUS_SUCCESS`：调用成功。
- `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`：`handle` 为空。
- `ACLBLAS_STATUS_INVALID_VALUE`：数组为空、维度/leading dimension/group size 非法，或任务规模溢出。
- `ACLBLAS_STATUS_NOT_SUPPORTED`：数据类型与 `computeType` 组合不受支持。
- `ACLBLAS_STATUS_ALLOC_FAILED`：设备侧 tiling 或 workspace 分配失败。
- `ACLBLAS_STATUS_EXECUTION_FAILED`：设备核心信息查询或 kernel 执行失败。

## 编译与测试

```bash
bash build.sh --soc=ascend950 --ops=gemm_grouped_batched_ex
bash build.sh --soc=ascend950 --ops=gemm_grouped_batched_ex --run
```
