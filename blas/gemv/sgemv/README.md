## Sgemv 算子实现

## 概述

BLAS Sgemv 算子实现。

Sgemv（Single-precision General Matrix-Vector multiplication）算子实现了单精度浮点通用矩阵与向量的乘法运算，是 BLAS Level 2 核心算子之一。针对 Ascend 950（arch35 / DAV_3510）架构，采用 SIMT 编程模型实现，支持不转置、转置和共轭转置三种模式，以及任意 incx/incy 步长（含负步长）。

## 支持的产品

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- |:-------:|
| Ascend 950PR / Ascend 950DT                                  |    ✓    |

## 目录结构介绍

```
blas/gemv/sgemv/
├── README.md                       // 说明文档
└── arch35/
    ├── sgemv_host.cpp              // Host 侧实现（参数校验、Tiling 计算、Kernel 调用）
    ├── sgemv_kernel.cpp            // Kernel 侧实现（SgemvNGm/SgemvTGm/SgemvNUb/SgemvTUb SIMT VF 函数）
    └── sgemv_tiling_data.h         // Tiling 数据结构（Host 和 Kernel 共用）
```

## 算子描述

- 算子功能：

sgemv 算子实现了单精度浮点矩阵 A 与向量 x 的乘法运算，并加到向量 y 上。对应的数学表达式为：

```
y = alpha * op(A) * x + beta * y
```

其中 `op(A)` 可以是：
- `A`（不转置，trans = N）：`y[i] = alpha * Σ(A[i,j] * x[j]) + beta * y[i]`，i = 0..m-1
- `A^T`（转置，trans = T）：`y[j] = alpha * Σ(A[i,j] * x[i]) + beta * y[j]`，j = 0..n-1
- `A^H`（共轭转置，trans = C，实数矩阵等价于转置）

矩阵 A 采用列主序（column-major）存储。

- 对应的接口为：

```cpp
aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle,
    aclblasOperation_t trans,
    int m,
    int n,
    const float *alpha,
    const float *A,
    int lda,
    const float *x,
    int incx,
    const float *beta,
    float *y,
    int incy);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">sgemv 参数说明</td>
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
      <td align="center">ops-blas 库上下文句柄。</td>
   </tr>
   <tr>
      <td align="center">trans</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">矩阵操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，实数等价于转置）。</td>
   </tr>
   <tr>
      <td align="center">m</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">矩阵 A 的行数，m ≥ 0。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">矩阵 A 的列数，n ≥ 0。</td>
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
      <td align="center">列主序 m × n 矩阵（float 数组），维度为 lda × n。当 m > 0 且 n > 0 时不可为 nullptr。</td>
   </tr>
   <tr>
      <td align="center">lda</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">矩阵 A 的主维（leading dimension），lda ≥ max(1, m)。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">Device</td>
      <td align="center">in</td>
      <td align="center">输入向量（float 数组），trans=N 时逻辑长度 n，trans=T/C 时逻辑长度 m。当对应维度 > 0 时不可为 nullptr。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">向量 x 的元素步长，incx ≠ 0。支持正负值。</td>
   </tr>
   <tr>
      <td align="center">beta</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">标量 β（float），不可为 nullptr。若 beta == 0，则 y 的输入值不被使用。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">Device</td>
      <td align="center">in/out</td>
      <td align="center">输入/输出向量（float 数组），trans=N 时逻辑长度 m，trans=T/C 时逻辑长度 n。当对应维度 > 0 时不可为 nullptr。</td>
   </tr>
   <tr>
      <td align="center">incy</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">向量 y 的元素步长，incy ≠ 0。支持正负值。</td>
   </tr>
</table>

- 算子规格：

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Sgemv</td></tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">lda × n</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">x</td><td align="center">trans=N: n, trans=T/C: m</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">trans=N: m, trans=T/C: n</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">trans=N: m, trans=T/C: n</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sgemv_kernel</td></tr>
  </table>

- 算子实现：

    根据 `trans` 参数和步长条件选择不同的计算路径，采用 GM/UB 双路径 dispatch 策略：

    - **trans=N**：每个 SIMT 线程处理若干输出行（grid-stride 循环）。当 `incx==1` 时走 UB 路径（`SgemvNUb`），先将 x 向量缓存到 `__ubuf__` 共享内存，每线程从 UB 读取 x 与 A 的行做点积；否则走 GM 路径（`SgemvNGm`），直接从全局内存读取 x。
    - **trans=T/C**：每个 SIMT 线程处理若干输出列（grid-stride 循环）。当 `incx==1` 时走 UB 路径（`SgemvTUb`），缓存 x 到 `__ubuf__`；否则走 GM 路径（`SgemvTGm`）。

    多核并行策略：按输出向量维度（trans=N 时为 m，trans=T 时为 n）均匀分配到多个 AIV Core，核数由 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 动态获取。

- 调用实现：
    使用内核调用符 `<<<>>>` 调用核函数。

## 调用示例

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    // 1. 初始化 ACL 和 ops-blas 句柄
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);

    // 2. 设置矩阵和向量参数
    int m = 256;       // 矩阵 A 行数
    int n = 128;       // 矩阵 A 列数
    int lda = m;       // 列主序，lda >= m
    int incx = 1;      // x 步长
    int incy = 1;      // y 步长
    float alpha = 1.0f;
    float beta = 0.0f;

    // 3. 分配设备内存并初始化数据
    float *d_A = nullptr;
    float *d_x = nullptr;
    float *d_y = nullptr;
    size_t sizeA = (size_t)lda * n * sizeof(float);
    size_t sizeX = (size_t)n * sizeof(float);   // trans=N: x 长度 n
    size_t sizeY = (size_t)m * sizeof(float);   // trans=N: y 长度 m

    aclrtMalloc((void**)&d_A, sizeA, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&d_x, sizeX, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&d_y, sizeY, ACL_MEM_MALLOC_HUGE_FIRST);

    // ... 初始化 d_A, d_x, d_y 数据（通过 aclrtMemcpy H2D）...

    // 4. 调用 sgemv: y = alpha * A * x + beta * y (trans=N)
    aclblasStatus_t status = aclblasSgemv(
        handle,
        ACLBLAS_OP_N,    // 不转置
        m, n,
        &alpha,
        d_A, lda,
        d_x, incx,
        &beta,
        d_y, incy);

    if (status == ACLBLAS_STATUS_SUCCESS) {
        // 5. 将结果从设备拷贝到主机
        float *h_y = new float[m];
        aclrtMemcpy(h_y, sizeY, d_y, sizeY, ACL_MEMCPY_DEVICE_TO_HOST);
        // ... 使用 h_y 结果 ...
        delete[] h_y;
    }

    // 6. 释放资源
    aclrtFree(d_A);
    aclrtFree(d_x);
    aclrtFree(d_y);
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

  - 默认路径，root 用户安装 CANN 软件包
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - 默认路径，非 root 用户安装 CANN 软件包
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - 指定路径 install_path，安装 CANN 软件包
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- 编译算子

  ```bash
  bash build.sh --ops=sgemv --soc=ascend950
  ```

- 编译并运行测试

  ```bash
  bash build.sh --ops=sgemv --soc=ascend950 --run
  ```

  其中 `--soc` 为**可选**参数，用于指定目标硬件平台：

  | 产品 | `--soc` 取值 |
  |------|--------------|
  | Ascend 950PR / Ascend 950DT | `ascend950` |

  执行结果如下，说明精度对比成功：
  ```
  [PASS] sgemv_test
  ```
