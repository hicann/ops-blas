## stbmv算子实现

## 概述

BLAS stbmv算子实现。

stbmv(Triangular Band Matrix-Vector Multiplication)算子实现了三角带状矩阵与向量的乘法运算，是BLAS基础线性代数库中的核心算子之一。

该算子支持上三角和下三角矩阵，支持转置和共轭转置操作，支持单位对角线和非单位对角线。针对arch22（Atlas A2/A3）和arch35（Ascend 950）分别进行了实现和优化。

## 产品支持情况

| 产品                                                         |  是否支持 | 架构 |
| :----------------------------------------------------------- |:-------:|:----:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ✓    | arch35 |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ✓    | arch22 |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    ✓    | arch22 |

## 目录结构介绍

```
blas/tbmv/stbmv/
├── README.md                              // 说明文档
├── arch35/
│   ├── stbmv_common.h                     // Tiling 数据结构与 kernel 声明
│   ├── stbmv_host.cpp                     // Host 侧实现
│   ├── stbmv_fallback_kernel.cpp          // SIMT 通用 kernel（compute/copy/diag）
│   └── stbmv_simd_fastpath_kernel.cpp     // SIMD 快速路径 kernel
└── arch22/
    ├── stbmv_host.cpp                     // Host 侧实现
    └── stbmv_kernel.cpp                   // Kernel 侧实现
```

测试代码位于 `test/tbmv/stbmv/`：

```
test/tbmv/stbmv/
├── CMakeLists.txt                         // 编译工程文件
├── stbmv_param.h                          // 参数结构体（继承 BlasTestParamBase）
├── stbmv_golden.h                         // CPU golden（签名与 BLAS API 一致）
├── arch35/
│   ├── stbmv_npu_wrapper.h                // NPU wrapper（封装 aclrtMalloc/H2D/kernel/D2H/free）
│   ├── stbmv_test.cpp                     // 精度测试（GTest 入口）
│   └── stbmv_test.csv                     // 精度测试用例表
└── arch22/
    └── stbmv_test.cpp                     // 精度测试
```

## 算子描述

- 算子功能：  
stbmv算子实现了三角带状矩阵乘以向量的运算。对应的数学表达式为：  
```
x = op(A) * x    （arch35，原地覆盖）
y = A * x        （arch22，输入输出分离）
```
A为n阶三角带状矩阵（上三角或下三角），半带宽为k，x/y为n维向量。arch35接口中op(A)可以是A、A的转置或A的共轭转置（实数下同转置）；arch22接口固定为不转置。

矩阵A采用列主序带状存储，需要`lda * n`个元素，其中`lda >= k + 1`。对于上三角矩阵，元素`A(i,j)`存储在位置`k + i - j + j * lda`（`max(0, j-k) <= i <= j`）；对于下三角矩阵，元素`A(i,j)`存储在位置`i - j + j * lda`（`j <= i <= min(n-1, j+k)`）。

**arch22 接口（Atlas A2/A3）：**

```
aclblasStatus_t aclblasStbmv_legacy(
    aclblasHandle_t handle,
    const float *a,
    const int64_t lda,
    const float *x,
    float *y,
    const int64_t n,
    const int64_t k,
    const int64_t incx);
```

| Param. | Memory | in/out | 含义 |
|:------:|:------:|:------:|------|
| handle |        | in     | aclBLAS 库上下文句柄。 |
| a      |   /    | in     | 下三角带状矩阵 float 数组，维度为 lda x n。 |
| lda    |        | in     | 矩阵 a 存储的主维长度，lda >= k + 1。 |
| x      |   /    | in     | float 输入向量，包含 n 个元素。 |
| y      |   /    | out    | float 输出向量，包含 n 个元素。 |
| n      |        | in     | 矩阵 A 的行数和列数。 |
| k      |        | in     | 下三角带状矩阵的半带宽。 |
| incx   |        | in     | x 中连续元素之间的步长。 |

> `aclblasStbmv_legacy` 为早期贡献接口，参数签名与标准 BLAS tbmv 不一致（固定下三角、不转置、非单位对角线，输入输出分离）。该接口后续可能调整或删除，建议新代码使用标准 `aclblasStbmv` 接口。

**arch35 接口（Ascend 950）：**

```
aclblasStatus_t aclblasStbmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    int k,
    const float *A,
    int lda,
    float *x,
    int incx);
```

| Param. | Memory | in/out | 含义 |
|:------:|:------:|:------:|------|
| handle |        | in     | aclBLAS 库上下文句柄。 |
| uplo   |        | in     | 矩阵填充类型：ACLBLAS_UPPER(上三角)、ACLBLAS_LOWER(下三角)。 |
| trans  |        | in     | 矩阵操作类型：ACLBLAS_OP_N(不转置)、ACLBLAS_OP_T(转置)、ACLBLAS_OP_C(共轭转置，实数下同T)。 |
| diag   |        | in     | 对角线类型：ACLBLAS_NON_UNIT(非单位对角线)、ACLBLAS_UNIT(单位对角线，对角元素视为1)。 |
| n      |        | in     | 三角带状矩阵 A 的行数和列数。 |
| k      |        | in     | 三角带状矩阵的半带宽。 |
| A      | device | in     | 三角带状矩阵 float 数组，维度为 lda x n。 |
| lda    |        | in     | 矩阵 A 存储的主维长度，lda >= k + 1。 |
| x      | device | in/out | float 向量，包含 n 个元素。输入为原始向量，输出为计算结果（原地覆盖）。 |
| incx   |        | in     | x 中连续元素之间的步长，不可为0。 |

> arch35 接口支持完整的 uplo/trans/diag 参数组合，采用原地覆盖模式（x 同时作为输入和输出）。


- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">stbmv</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">lda * N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">x (arch35) / y (arch22)</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">核函数名</td><td colspan="6" align="center">arch22: tbmv_kernel</td></tr>
  <tr><td colspan="6" align="center">arch35 SIMD fastpath: stbmv_column_simd_fast_kernel / stbmv_simd_fast_copy_kernel</td></tr>
  <tr><td colspan="6" align="center">arch35 SIMT fallback: stbmv_compute_kernel / stbmv_copy_kernel / stbmv_diag_kernel</td></tr>
  </table>

- 算子实现：

  - **arch35**（Ascend 950）：

    采用双路径实现。Host侧根据矩阵规模和带宽自动选择SIMD快速路径或SIMT通用路径：

    - **SIMD快速路径**（`stbmv_simd_fastpath_kernel.cpp`）：适用于满足SIMD对齐条件的场景。`stbmv_column_simd_fast_kernel`按列方向并行计算矩阵向量乘积，将中间结果写入workspace；`stbmv_simd_fast_copy_kernel`将workspace中的连续结果按incx步长写回x向量。

    - **SIMT通用路径**（`stbmv_fallback_kernel.cpp`）：适用于任意参数组合。当k=0时走`stbmv_diag_kernel`直接处理纯对角线缩放；否则`stbmv_compute_kernel`通过SIMT多线程并行，每个线程负责计算输出向量的若干行，将中间结果写入workspace；`stbmv_copy_kernel`将结果按incx步长写回x向量。

- 调用实现：
    使用内核调用符<<<>>>调用核函数。
  - **arch22**：调用单个 tbmv_kernel。
  - **arch35**：根据路径选择依次调用对应的 compute kernel 和 copy kernel。

## 测试用例覆盖

### arch35（Ascend 950）

| 分组 | 用例数 | 覆盖场景 |
|------|--------|----------|
| L0 | 12 | 基础功能：upper/lower × non-transpose/transpose/conj-trans × unit/non-unit、k=0、非紧凑lda、负步长 |
| L1 | 4 | 边界场景：n=1大k、k>n、n=0、n=1 |
| GEN | 14 | 大规模压力/性能：k=0大对角(n=65536)、宽带宽(n=32768)、大步长(n=65536 incx=-2)、满带宽(n=4096)、非紧凑lda+incx组合 |
| INV | 7 | 非法参数：invalid uplo/trans/diag、n<0、k<0、lda<k+1、incx=0 |

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。
- 配置环境变量  
  请根据当前环境上CANN开发套件包的安装方式，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - 默认路径，非root用户安装CANN软件包
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - 指定路径install_path，安装CANN软件包
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- 样例执行
  ```bash
  bash build.sh --ops=stbmv --soc=ascend950 --run
  ```

  其中`--soc` 为**可选**参数，用于指定目标硬件平台（与上文「产品支持情况」对应），决定编译哪个架构的实现：

  | 产品 | `--soc` 取值 | 架构 |
  |------|----------------|:----:|
  | Ascend 950PR / Ascend 950DT | `ascend950` | arch35 |
  | Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | `ascend910_93` | arch22 |
  | Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | `ascend910b` | arch22 |

  执行结果如下，说明精度对比成功。
  ```bash
  [PASS] stbmv_test
  ```
