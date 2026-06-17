## Gemv算子实现

## 概述

BLAS Gemv算子实现，包含实数矩阵-向量乘法（Sgemv）和复数矩阵-向量乘法（Cgemv）。

Gemv（General Matrix-Vector multiplication）算子实现了通用矩阵与向量的乘法运算，是BLAS Level 2核心算子之一。

该算子包含以下接口：
- **aclblasSgemv**：单精度浮点矩阵-向量乘法，针对 arch35（Ascend 950）采用 SIMT 编程模型实现
- **aclblasCgemv**：复数矩阵-向量乘法，针对 arch22（Atlas A2/A3）实现

## 产品支持情况

| 产品                                                         |  是否支持 | 架构 |
| :----------------------------------------------------------- |:-------:|:----:|
| Ascend 950PR / Ascend 950DT                                  |    ✓    | arch35 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品                    |    ✓    | arch22 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品                    |    ✓    | arch22 |

## 目录结构介绍

```
blas/gemv/
├── README.md                       // 说明文档
├── arch22/
│   ├── ascblasCgemv_utils.h        // Cgemv 工具函数（arch22）
│   ├── cgemv_host.cpp              // Cgemv Host 侧实现（arch22）
│   ├── cgemv_no_trans_kernel.cpp   // Cgemv 不转置 Kernel（arch22）
│   └── cgemv_do_trans_kernel.cpp   // Cgemv 转置 Kernel（arch22）
└── arch35/
    ├── sgemv_host.cpp              // Sgemv Host 侧实现（arch35）
    ├── sgemv_kernel.cpp            // Sgemv Kernel 侧实现（arch35）
    └── sgemv_tiling_data.h         // Sgemv Tiling 数据结构（arch35）
```

## 算子描述

### Sgemv（单精度浮点矩阵-向量乘法）

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
    int m, int n,
    const float *alpha,
    const float *A, int lda,
    const float *x, int incx,
    const float *beta,
    float *y, int incy);
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
      <td align="center">矩阵 A 的行数，m >= 0。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">矩阵 A 的列数，n >= 0。</td>
   </tr>
   <tr>
      <td align="center">alpha</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">标量 alpha（float），不可为 nullptr。</td>
   </tr>
   <tr>
      <td align="center">A</td>
      <td align="center">Device</td>
      <td align="center">in</td>
      <td align="center">列主序 m x n 矩阵（float 数组），维度为 lda x n。</td>
   </tr>
   <tr>
      <td align="center">lda</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">矩阵 A 的主维（leading dimension），lda >= max(1, m)。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">Device</td>
      <td align="center">in</td>
      <td align="center">输入向量（float 数组），trans=N 时逻辑长度 n，trans=T/C 时逻辑长度 m。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">向量 x 的元素步长，incx != 0。支持正负值。</td>
   </tr>
   <tr>
      <td align="center">beta</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">标量 beta（float），不可为 nullptr。若 beta == 0，则 y 的输入值不被使用。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">Device</td>
      <td align="center">in/out</td>
      <td align="center">输入/输出向量（float 数组），trans=N 时逻辑长度 m，trans=T/C 时逻辑长度 n。</td>
   </tr>
   <tr>
      <td align="center">incy</td>
      <td align="center">Host</td>
      <td align="center">in</td>
      <td align="center">向量 y 的元素步长，incy != 0。支持正负值。</td>
   </tr>
</table>

- 算子规格：

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Sgemv</td></tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">lda x n</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">x</td><td align="center">trans=N: n, trans=T/C: m</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">trans=N: m, trans=T/C: n</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">trans=N: m, trans=T/C: n</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sgemv_kernel</td></tr>
  </table>

- 算子实现：

    根据 `trans` 参数和步长条件选择不同的计算路径，采用 GM/UB 双路径 dispatch 策略：

    - **trans=N**：每个 SIMT 线程处理若干输出行（grid-stride 循环）。当 `incx==1` 时走 UB 路径（`SgemvNUb`），先将 x 向量缓存到 `__ubuf__` 共享内存，每线程从 UB 读取 x 与 A 的行做点积；否则走 GM 路径（`SgemvNGm`），直接从全局内存读取 x。
    - **trans=T/C**：每个 SIMT 线程处理若干输出列（grid-stride 循环）。当 `incx==1` 时走 UB 路径（`SgemvTUb`），缓存 x 到 `__ubuf__`；否则走 GM 路径（`SgemvTGm`）。

    多核并行策略：按输出向量维度（trans=N 时为 m，trans=T 时为 n）均匀分配到多个 AIV Core。

### Cgemv（复数矩阵-向量乘法）

- 算子功能：

cgemv算子实现了复数矩阵A与向量x的乘法运算，并加到向量y上。对应的数学表达式为：
```
y = alpha * op(A) * x + beta * y
```
其中op(A)可以是：
- A（不转置，trans=N）
- A^T（转置，trans=T）
- A^H（共轭转置，trans=C）

复数乘法公式：`(a+bi) * (c+di) = (ac-bd) + (ad+bc)i`

- 对应的接口：
```cpp
int aclblasCgemv(aclblasHandle handle,
                  aclblasOperation trans,
                  const int64_t m, const int64_t n,
                  const std::complex<float> &alpha,
                  const std::complex<float> *A, const int64_t lda,
                  const std::complex<float> *x, const int64_t incx,
                  const std::complex<float> &beta,
                  std::complex<float> *y, const int64_t incy,
                  void *stream);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">cgemv 参数说明</td>
   </tr>
   <tr>
      <td rowspan="12" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">handle</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">ACL流handle，用于传入stream。</td>
   </tr>
   <tr>
      <td align="center">trans</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">矩阵操作类型：N=不转置，T=转置，C=共轭转置。</td>
   </tr>
   <tr>
      <td align="center">m</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵A的行数。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵A的列数。</td>
   </tr>
   <tr>
      <td align="center">alpha</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">复数标量alpha。</td>
   </tr>
   <tr>
      <td align="center">A</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">m x n复数矩阵。</td>
   </tr>
   <tr>
      <td align="center">lda</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵A的主维长度。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">向量x（长度取决于trans）。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">x中连续元素之间的步长。</td>
   </tr>
   <tr>
      <td align="center">beta</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">复数标量beta。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">向量y（长度取决于trans）。</td>
   </tr>
   <tr>
      <td align="center">incy</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">y中连续元素之间的步长。</td>
   </tr>
</table>

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Cgemv</td></tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">m x n</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  <tr><td align="center">x</td><td align="center">trans=N: n, trans=T/C: m</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">trans=N: m, trans=T/C: n</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cgemv_no_trans / cgemv_do_trans</td></tr>
  </table>

- 算子实现：

    根据是否转置，选择不同的kernel函数：
    - cgemv_no_trans：不转置情况，矩阵按列分块，每列与向量元素相乘后累加到y
    - cgemv_do_trans：转置情况，矩阵按行分块，每行与向量元素相乘后累加到y

    使用vreducev2进行虚实分离，使用vgather进行虚实合并，利用AsdopsBuffer进行乒乓缓冲优化。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 编译运行

在仓库根目录下执行如下步骤，编译并运行算子测试。

- 配置环境变量
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- 编译并执行测试
  ```bash
  bash build.sh --ops=sgemv --soc=ascend950 --run
  bash build.sh --ops=cgemv --run
  ```

  执行结果如下，说明精度对比成功。
  ```bash
  [PASS] sgemv_test
  [PASS] cgemv_test
  ```
