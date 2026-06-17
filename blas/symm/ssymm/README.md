## Ssymm 算子实现

## 概述

BLAS Ssymm 算子实现。

Ssymm（Single-precision Symmetric Matrix Multiplication）算子实现了单精度浮点对称矩阵与普通矩阵的乘法运算，是 BLAS Level 3 核心算子之一。针对 Ascend 910B（arch22）架构实现。

数学表达式：

- LEFT 模式：`C := alpha * A * B + beta * C`
- RIGHT 模式：`C := alpha * B * A + beta * C`

其中 A 为对称矩阵（仅存储上三角或下三角），B 和 C 为普通矩阵。

## 支持的产品

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- |:-------:|
| Ascend 910B / Ascend 910_93 (A2/A3 训练推理系列)              |    ✓    |

## 目录结构介绍

```
blas/symm/ssymm/
├── README.md                       // 说明文档
├── ssymm_common_host.h             // Host 侧公共头文件
├── ssymm_common_kernel.h           // Kernel 侧公共头文件
├── ssymm_common_types.h            // 公共类型定义
└── arch22/
    ├── ssymm_host.cpp              // Host 侧实现（参数校验、Tiling 计算、Kernel 调用）
    ├── ssymm_kernel.cpp            // Kernel 侧实现
    └── ssymm_kernel_fwd.h          // Kernel 前向声明
```

## 算子描述

- 算子功能：

ssymm 算子实现了对称矩阵 A 与普通矩阵 B 的乘法运算，并累加到矩阵 C 上。

当 `side = LEFT` 时 A 为 m×m 矩阵，B 和 C 为 m×n 矩阵。
当 `side = RIGHT` 时 A 为 n×n 矩阵，B 和 C 为 m×n 矩阵。

- 对应的接口为：

```cpp
aclblasStatus_t aclblasSsymm(
    aclblasHandle handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    int64_t m,
    int64_t n,
    const float *alpha,
    const float *A,
    int64_t lda,
    const float *B,
    int64_t ldb,
    const float *beta,
    float *C,
    int64_t ldc
);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">ssymm 参数说明</td>
   </tr>
   <tr>
      <td rowspan="13" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">handle</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">ACL-BLAS 句柄。</td>
   </tr>
   <tr>
      <td align="center">side</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">A 矩阵位置：ACLBLAS_SIDE_LEFT（左侧）或 ACLBLAS_SIDE_RIGHT（右侧）。</td>
   </tr>
   <tr>
      <td align="center">uplo</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">A 矩阵存储模式：ACLBLAS_LOWER（下三角）或 ACLBLAS_UPPER（上三角）。</td>
   </tr>
   <tr>
      <td align="center">m</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">矩阵 C 的行数，m ≥ 0。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">矩阵 C 的列数，n ≥ 0。</td>
   </tr>
   <tr>
      <td align="center">alpha</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">标量 α（float），不可为 nullptr。</td>
   </tr>
   <tr>
      <td align="center">A</td>
      <td align="center">Device</td>
      <td align="center">in</td>
      <td align="center">对称矩阵（float 数组），side=LEFT 时 m×m，side=RIGHT 时 n×n。</td>
   </tr>
   <tr>
      <td align="center">lda</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">矩阵 A 的主维，side=LEFT 时 lda ≥ max(1, m)，side=RIGHT 时 lda ≥ max(1, n)。</td>
   </tr>
   <tr>
      <td align="center">B</td>
      <td align="center">Device</td>
      <td align="center">in</td>
      <td align="center">m×n 普通矩阵（float 数组）。</td>
   </tr>
   <tr>
      <td align="center">ldb</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">矩阵 B 的主维，ldb ≥ max(1, n)。</td>
   </tr>
   <tr>
      <td align="center">beta</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">标量 β（float），不可为 nullptr。</td>
   </tr>
   <tr>
      <td align="center">C</td>
      <td align="center">Device</td>
      <td align="center">in/out</td>
      <td align="center">m×n 矩阵（float 数组），输入旧值，输出新值。</td>
   </tr>
   <tr>
      <td align="center">ldc</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">矩阵 C 的主维，ldc ≥ max(1, n)。</td>
   </tr>
</table>

- 算子规格：

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Ssymm</td></tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">side=LEFT: m×m, side=RIGHT: n×n</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">B</td><td align="center">m×n</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">C</td><td align="center">m×n</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">ssymm_kernel</td></tr>
  </table>

- 算子实现：

  采用 OptimizedLeft / OptimizedRight / GenericFallback 三路 backend 分派策略。

  - **OptimizedLeft / LeftCube**：LEFT 模式 Cube 优化路径，处理常规 shape 的 LEFT 下三角/上三角。
  - **OptimizedRight / RightCube**：RIGHT 模式 Cube 优化路径，处理常规 shape 的 RIGHT 下三角/上三角。
  - **GenericFallback / GenericKernel**：通用回退路径，处理小 shape、padded ld、异常入参等场景。

  多核并行：Block Dim 固定为 8（vector core），LEFT 路径为 5 阶段流水线（clear_partial → pack → dense → accum → postprocess），RIGHT 路径为 3 阶段流水线（pack → dense → accum）。

- 调用实现：

  使用内核调用符 `<<<>>>` 调用核函数，通过 `BuildSsymmExecutionPlan` 选择 backend 路径。

- 支持的数据类型：FP32（单精度浮点数）

- 精度要求：

  | 精度标准 | MARE 阈值 | MERE 阈值 |
  |---------|-----------|-----------|
  | 浮点计算类社区标准（单标杆） | ≤ 10 · 2⁻¹³ | ≤ 2⁻¹³ |

## 调用示例

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    // 1. 初始化 ACL 和 ops-blas 句柄
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle handle = nullptr;
    aclblasCreate(&handle);

    // 2. 设置矩阵参数
    int64_t m = 256;
    int64_t n = 256;
    int64_t lda = m, ldb = n, ldc = n;
    float alpha = 1.25f, beta = 0.5f;

    // 3. 分配设备内存
    float *d_A, *d_B, *d_C;
    aclrtMalloc((void**)&d_A, lda * m * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&d_B, ldb * n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&d_C, ldc * n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    // ... 初始化 d_A, d_B, d_C 数据 ...

    // 4. 调用 ssymm: C = alpha * A * B + beta * C (LEFT, LOWER)
    aclblasStatus_t status = aclblasSsymm(
        handle,
        ACLBLAS_SIDE_LEFT,
        ACLBLAS_LOWER,
        m, n,
        &alpha, d_A, lda,
        d_B, ldb,
        &beta, d_C, ldc);

    if (status == ACLBLAS_STATUS_SUCCESS) {
        // ... 使用结果 ...
    }

    // 5. 释放资源
    aclrtFree(d_A);
    aclrtFree(d_B);
    aclrtFree(d_C);
    aclblasDestroy(handle);
    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```

## 编译运行

在仓库根目录下执行如下步骤，编译并运行算子测试。

- 配置环境变量

  请根据当前环境上 CANN 开发套件包的安装方式，选择对应配置环境变量的命令。

  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- 编译算子

  ```bash
  bash build.sh --ops=ssymm --soc=ascend910b
  ```

- 编译并运行测试

  ```bash
  bash build.sh --ops=ssymm --soc=ascend910b --run
  ```

  执行结果如下，说明精度对比成功：

  ```
  [Success] Case accuracy is verification passed.
  ```
