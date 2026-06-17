## Trmv算子实现

## 概述

BLAS Trmv算子实现，包含实数三角矩阵-向量乘法（Strmv）和复数三角矩阵-向量乘法（Ctrmv）。

Trmv（Triangular Matrix-Vector Multiplication）算子实现了三角矩阵与向量的乘法运算，是BLAS基础线性代数库中的Level 2核心算子之一。

该算子包含以下接口：
- **aclblasStrmv**：实数三角矩阵-向量乘法，支持 arch22 和 arch35 两种架构
- **aclblasCtrmv**：复数三角矩阵-向量乘法，支持 arch22 架构

## 产品支持情况

| 产品                                                         |  是否支持 | 架构 |
| :----------------------------------------------------------- |:-------:|:----:|
| Ascend 950PR / Ascend 950DT                                  |    ✓    | arch35 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品                    |    ✓    | arch22 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品                    |    ✓    | arch22 |

## 目录结构介绍

```
blas/trmv/
├── README.md                       // 说明文档
├── arch22/
│   ├── ctrmv_host.cpp              // Ctrmv Host 侧实现（arch22）
│   ├── ctrmv_kernel.cpp            // Ctrmv Kernel 侧实现（arch22）
│   ├── strmv_host.cpp              // Strmv Host 侧实现（arch22）
│   └── strmv_kernel.cpp            // Strmv Kernel 侧实现（arch22）
└── arch35/
    ├── strmv_common.h              // Strmv Tiling 数据结构与 kernel 声明（arch35）
    ├── strmv_host.cpp              // Strmv Host 侧实现（arch35）
    └── strmv_kernel.cpp            // Strmv Kernel 侧实现（arch35）
```

## 算子描述

### Strmv（实数三角矩阵-向量乘法）

- 算子功能：

strmv算子实现了三角矩阵乘以向量的运算，结果覆盖到输入向量x中。对应的数学表达式为：
```
x = op(A) * x
```
A为n阶三角矩阵（上三角或下三角），x为n维向量，op(A)可以是A、A的转置或A的共轭转置（实数下同转置）。

矩阵A采用列主序全矩阵存储，需要`lda * n`个元素，其中`lda >= n`。

- 对应的接口为：
```cpp
aclblasStatus_t aclblasStrmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    const float *A,
    int lda,
    float *x,
    int incx);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">strmv 参数说明</td>
   </tr>
   <tr>
      <td rowspan="10" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">handle</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">aclBLAS 库上下文句柄。</td>
   </tr>
   <tr>
      <td align="center">uplo</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵填充类型：ACLBLAS_UPPER(上三角)、ACLBLAS_LOWER(下三角)。</td>
   </tr>
   <tr>
      <td align="center">trans</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵操作类型：ACLBLAS_OP_N(不转置)、ACLBLAS_OP_T(转置)、ACLBLAS_OP_C(共轭转置，实数下同T)。</td>
   </tr>
   <tr>
      <td align="center">diag</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">对角线类型：ACLBLAS_NON_UNIT(非单位对角线)、ACLBLAS_UNIT(单位对角线，对角元素视为1)。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">三角矩阵 A 的行数和列数。</td>
   </tr>
   <tr>
      <td align="center">A</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">三角矩阵 float 数组，维度为 lda x n。</td>
   </tr>
   <tr>
      <td align="center">lda</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵 A 存储的主维长度，lda >= n。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">float 向量，包含 n 个元素。输入为原始向量，输出为计算结果（原地覆盖）。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">x 中连续元素之间的步长，不可为0。</td>
   </tr>
</table>

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">Strmv</td></tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">lda * N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">核函数名</td><td colspan="6" align="center">arch22: strmv</td></tr>
  <tr><td colspan="6" align="center">arch35: strmv_compute_kernel / strmv_copy_kernel</td></tr>
  </table>

- 算子实现：

  - **arch22**（Atlas A2/A3）：

    采用单kernel实现。Host侧将矩阵按`M0=128`分块，构造128x128的上/下三角掩码矩阵并拷贝至Device。Kernel侧每个AI Core负责一个行分块：将矩阵子块从GM搬运到UB，通过掩码矩阵屏蔽无效三角区域元素，处理单位对角线；将向量x从GM搬运到UB（incx=1时直接拷贝，否则按步长抽取）；通过`vaxpy`/`vmla`+`vcadd`完成矩阵向量乘加；结果写回workspace，最后通过跨核同步和步长写回将结果覆盖到x。

  - **arch35**（Ascend 950）：

    采用两阶段SIMT kernel实现。第一阶段`strmv_compute_kernel`通过SIMT多线程并行，每个线程负责计算输出向量的若干行：根据uplo/trans确定每行的有效列范围，从GM读取矩阵A和向量x的元素，在寄存器中完成乘加累加，将中间结果写入workspace。第二阶段`strmv_copy_kernel`将workspace中的连续结果按incx步长写回x向量。通过模板参数`<UPLO_IS_UPPER, TRANS_IS_N, DIAG_IS_UNIT>`编译期分发8种组合，消除运行时分支。

### Ctrmv（复数三角矩阵-向量乘法）

- 算子功能：

Ctrmv算子实现了复数三角矩阵与向量的乘法。对应的数学表达式为：
```
x = op(A) * x
```
其中：
- op(A) = A，当 trans = N
- op(A) = A^T，当 trans = T
- op(A) = A^H，当 trans = C（共轭转置）
- A为n×n的复数三角矩阵（上三角或下三角）
- x为长度为n的复数向量

复数乘法公式：(a + bi) * (c + di) = (ac - bd) + (ad + bc)i

- 对应的接口为：
```cpp
int aclblasCtrmv(aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
                 aclblasDiagType_t diag, int64_t n,
                 const float *A, int64_t lda, float *x, int64_t incx);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">ctrmv 参数说明</td>
   </tr>
   <tr>
      <td rowspan="9" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">handle</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">BLAS句柄，用于传入stream。</td>
   </tr>
   <tr>
      <td align="center">uplo</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">指定矩阵A的上三角或下三角部分。ACLBLAS_UPPER或ACLBLAS_LOWER。</td>
   </tr>
   <tr>
      <td align="center">trans</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">指定对矩阵A的操作类型。ACLBLAS_OP_N(不转置)、ACLBLAS_OP_T(转置)、ACLBLAS_OP_C(共轭转置)。</td>
   </tr>
   <tr>
      <td align="center">diag</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">指定对角线元素是否为单位元。ACLBLAS_UNIT(单位对角线)或ACLBLAS_NON_UNIT(非单位对角线)。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵A的阶数，即向量的长度。</td>
   </tr>
   <tr>
      <td align="center">A</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">n×lda的复数矩阵，存储为2×n×lda个float。</td>
   </tr>
   <tr>
      <td align="center">lda</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵A的主维度。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">复数向量，长度为n，存储为2×n×incx个float。既是输入也是输出。</td>
   </tr>
</table>

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Ctrmv</td></tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">n × n</td><td align="center">complex&lt;float&gt;</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">x</td><td align="center">n</td><td align="center">complex&lt;float&gt;</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">ctrmv_kernel</td></tr>
  </table>

- 算子实现：

    将输入数据从A和x的GM地址分块搬运到UB，进行复数三角矩阵-向量乘法计算后再将结果从workspace回写到x所在的GM地址。支持上三角和下三角模式，支持不转置、转置和共轭转置操作。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

- 约束说明：
  - n的取值范围为[1, 8192]
  - 仅支持complex<float>数据类型
  - incx > 0
  - lda > 0

## 编译运行

在仓库根目录下执行如下步骤，编译并运行算子测试。

- 配置环境变量
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- 编译并执行测试
  ```bash
  bash build.sh --ops=strmv --soc=ascend950 --run
  bash build.sh --ops=ctrmv --run
  ```

  其中`--soc` 为**可选**参数，用于指定目标硬件平台：

  | 产品 | `--soc` 取值 | 架构 |
  |------|----------------|:----:|
  | Ascend 950PR / Ascend 950DT | `ascend950` | arch35 |
  | Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | `ascend910_93` | arch22 |
  | Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | `ascend910b` | arch22 |

  执行结果如下，说明精度对比成功。
  ```bash
  [PASS] strmv_test
  [PASS] ctrmv_test
  ```
