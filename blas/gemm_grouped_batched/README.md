## GemmGroupedBatched 算子实现

## 概述

BLAS GemmGroupedBatched 算子实现。

GemmGroupedBatched（分组批量矩阵乘法）实现了对多个分组内各批次矩阵独立进行 GEMM 运算，是 BLAS Level 3 核心算子之一。针对 Ascend 950（arch35）架构，当前仅保留 S（FP32）接口，并在 NPU 上完成计算。

数学表达式：

```
C[i] = alpha[g] × op(A[i]) × op(B[i]) + beta[g] × C[i]
```

其中 `i` 属于分组 `g`，`op(A)` / `op(B)` 由 transa / transb 决定：
- `N`：不转置
- `T`：转置

所有矩阵采用 **Column-Major**（列主序）存储。

## 支持的产品

| 产品                                 | 是否支持 |
| :----------------------------------- | :------: |
| Ascend 950PR / Ascend 950DT          |    ✓     |

## 目录结构介绍

```
blas/gemm_grouped_batched/
├── README.md                               // 说明文档
├── CMakeLists.txt                          // 构建配置
└── arch35/
    ├── gemm_grouped_batched_host.cpp       // Host 侧实现（参数校验、Tiling 计算、Kernel 调用）
    ├── gemm_grouped_batched_kernel.cpp     // Kernel 侧实现（AIV SIMD membase 模式批量运算）
    └── gemm_grouped_batched_tiling_data.h  // Tiling 数据结构（Host 和 Kernel 共用）
```

## 算子描述

- 算子功能：

  对 groupCount 个分组中的全部 batch 矩阵独立执行 GEMM 运算。每个分组 g 可拥有不同的 (m, n, k, transa, transb, alpha, beta, lda, ldb, ldc) 参数，以及 groupSize[g] 个批次。

- 接口：

  ```cpp
  // S 类型：FP32，NPU 执行
  aclblasStatus_t aclblasSgemmGroupedBatched(
      aclblasHandle_t handle, int groupCount,
      const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray,
      const int* mArray, const int* nArray, const int* kArray,
      const float* alphaArray, const float* const* Aarray, const int* ldaArray,
      const float* const* Barray, const int* ldbArray,
      const float* betaArray, float* const* Carray, const int* ldcArray,
      const int* groupSizeArray);
  ```

- 参数说明：

  | Param.          | Memory | in/out | 含义 |
  | :-------------- | :----- | :----: | :--- |
  | handle          | Host   | in     | ops-blas 库上下文句柄 |
  | groupCount      | Host   | in     | 分组数量 |
  | transaArray     | Host   | in     | 每组的 A 矩阵操作类型数组。当前 S 类型支持 N/T |
  | transbArray     | Host   | in     | 每组的 B 矩阵操作类型数组。当前 S 类型支持 N/T |
  | mArray          | Host   | in     | 每组的输出行数数组 |
  | nArray          | Host   | in     | 每组的输出列数数组 |
  | kArray          | Host   | in     | 每组的内积维度数组 |
  | alphaArray      | Host   | in     | 每组的标量乘数数组（S: float） |
  | Aarray          | Host   | in     | A 矩阵指针数组（Host 侧数组，元素为 Device 矩阵地址），长度 = sum(groupSize[g]) |
  | ldaArray        | Host   | in     | 每组的 A 矩阵 leading dimension 数组 |
  | Barray          | Host   | in     | B 矩阵指针数组（Host 侧数组，元素为 Device 矩阵地址），长度 = sum(groupSize[g]) |
  | ldbArray        | Host   | in     | 每组的 B 矩阵 leading dimension 数组 |
  | betaArray       | Host   | in     | 每组的标量乘数数组（类型与 alphaArray 一致） |
  | Carray          | Host   | in/out | C 矩阵指针数组（Host 侧数组，元素为 Device 矩阵地址），长度 = sum(groupSize[g]) |
  | ldcArray        | Host   | in     | 每组的 C 矩阵 leading dimension 数组 |
  | groupSizeArray  | Host   | in     | 每组的批次数量数组 |

- 算子规格：

  <table>
  <tr><td rowspan="1" align="center">算子类型</td><td colspan="4" align="center">SgemmGroupedBatched（FP32）</td></tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">storage order</td></tr>
  <tr><td align="center">A[i]</td><td align="center">lda × (trans=N: k, trans=T: m)</td><td align="center">float</td><td align="center">Column-Major</td></tr>
  <tr><td align="center">B[i]</td><td align="center">ldb × (trans=N: n, trans=T: k)</td><td align="center">float</td><td align="center">Column-Major</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">C[i]</td><td align="center">ldc × n</td><td align="center">float</td><td align="center">Column-Major</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">gemm_grouped_batched_kernel</td></tr>
  </table>

## 支持规格

| 项目 | 内容 |
|------|------|
| 数据类型 | S (FP32) — NPU 执行 |
| 目标芯片 | Ascend 950 |
| 目标架构 | arch35 |
| 存储顺序 | Column-Major |
| 编程模型 | AIV SIMD membase |
| trans 支持 | S: N, T（共 4 种组合：NN / NT / TN / TT） |

### 数据类型执行路径说明

| 数据类型 | 执行路径 | 说明 |
| :------- | :------- | :--- |
| S (FP32) | NPU | Kernel 采用 AIV SIMD membase 模式，多核并行处理各 batch |

## 算子实现

- **S 类型（NPU Kernel）**：
  - transa=N 时采用列逐步累加（ColumnWise）模式：逐列从 A/B 读取 tile，向量乘加到 C tile
  - transa=T 时采用行点积（DotProduct）模式：逐行向量点积后 ReduceSum 写入 C 元素
  - transb 标志控制 B 矩阵的拷贝与访问方式（`CopyInB` / 计算循环内按 N/T 分支读取）
  - TT（transa=T, transb=T）与其他转置组合一样，由 Kernel 按 transa/transb 标志走转置拷贝 + 对应计算路径，Host 侧不做 A/B 交换
  - alpha=0 或 k=0 时走 BetaOnly 路径：仅执行 C = beta × C
  - 多核并行：总 batch 数均匀分配到各 Vector Core


## 精度标准

| 数据类型 | 精度指标 | 验证标准 |
| :------- | :------- | :------- |
| S (FP32) | MERE < 2⁻¹³ (≈0.000122) | MERE &lt; Threshold，MARE &lt; 10×Threshold |

## 编译运行

- 环境配置
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- 编译算子
  ```bash
  bash build.sh --ops=gemm_grouped_batched --soc=ascend950
  ```

- 编译并运行测试
  ```bash
  bash build.sh --ops=gemm_grouped_batched --soc=ascend950 --run
  ```

## 调用示例

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

// ── S 类型调用示例 ──

// 初始化 ACL 和 handle
aclInit(nullptr);
aclrtSetDevice(0);
aclblasHandle_t handle = nullptr;
aclblasCreateHandle(&handle);

// 定义 2 个分组
int groupCount = 2;

// 分组 0: 4×4 GEMM, NN, 3 个 batch
// 分组 1: 8×8 GEMM, TN, 2 个 batch
aclblasOperation_t transaArray[] = {ACLBLAS_OP_N, ACLBLAS_OP_T};
aclblasOperation_t transbArray[] = {ACLBLAS_OP_N, ACLBLAS_OP_N};
int mArray[]      = {4, 8};
int nArray[]      = {4, 8};
int kArray[]      = {4, 4};
float alphaArray[] = {1.0f, 1.0f};
float betaArray[]  = {0.0f, 1.0f};
int ldaArray[]     = {4, 4};   // 分组0: trans=N, lda≥m=4; 分组1: trans=T, lda≥k=4
int ldbArray[]     = {4, 4};
int ldcArray[]     = {4, 8};
int groupSizeArray[] = {3, 2};

// A/B/C 指针数组（共 5 个 batch，Host 侧数组），元素指向 Device 上的列主序矩阵
const float* Aarray[5];
const float* Barray[5];
float*       Carray[5];

aclblasStatus_t ret = aclblasSgemmGroupedBatched(
    handle, groupCount,
    transaArray, transbArray, mArray, nArray, kArray,
    alphaArray, Aarray, ldaArray,
    Barray, ldbArray, betaArray,
    Carray, ldcArray, groupSizeArray);

// 清理
aclblasDestroyHandle(handle);
aclrtResetDevice(0);
aclFinalize();
```

## 约束与限制

- 矩阵存储顺序为 **Column-Major**，与部分其他 BLAS 算子的 Row-Major 不同
- 当前仅保留 S 类型接口；transa/transb 各支持 N/T，组合为 NN / NT / TN / TT 四种
- `Aarray`/`Barray`/`Carray` 为 **Host 侧**指针数组，元素指向 Device 上的矩阵数据；不可传入 Device 侧指针数组
- alpha=0 时跳过矩阵乘法，仅执行 C = beta × C
- k=0 时等效为 C = beta × C
- lda/ldb/ldc 需满足 BLAS 标准约束（lda ≥ max(1, trans=N 时为 m, trans=T 时为 k)）