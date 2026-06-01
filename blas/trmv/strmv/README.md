## strmv算子实现

## 概述

BLAS strmv算子实现。

strmv(Triangular Matrix-Vector Multiplication)算子实现了三角矩阵与向量的乘法运算，是BLAS基础线性代数库中的核心算子之一。

该算子支持上三角和下三角矩阵，支持转置和共轭转置操作，支持单位对角线和非单位对角线。针对arch22（Atlas A2/A3）和arch35（Ascend 950）分别进行了实现和优化。

## 产品支持情况

| 产品                                                         |  是否支持 | 架构 |
| :----------------------------------------------------------- |:-------:|:----:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ✓    | arch35 |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ✓    | arch22 |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    ✓    | arch22 |

## 目录结构介绍

```
blas/trmv/strmv/
├── README.md                       // 说明文档
├── arch35/
│   ├── strmv_common.h              // Tiling 数据结构与 kernel 声明
│   ├── strmv_host.cpp              // Host 侧实现
│   └── strmv_kernel.cpp           // Kernel 侧实现
└── arch22/
    ├── strmv_host.cpp              // Host 侧实现
    └── strmv_kernel.cpp           // Kernel 侧实现
```

测试代码位于 `test/trmv/strmv/`：

```
test/trmv/strmv/
├── CMakeLists.txt                  // 编译工程文件
├── strmv_param.h                   // 参数结构体（继承 BlasTestParamBase）
├── strmv_golden.h                  // CPU golden（签名与 BLAS API 一致）
├── arch35/
│   ├── strmv_npu_wrapper.h         // NPU wrapper（封装 aclrtMalloc/H2D/kernel/D2H/free）
│   ├── strmv_test.cpp              // 精度测试（GTest 入口）
│   └── strmv_test.csv              // 精度测试用例表
└── arch22/
    └── strmv_test.cpp              // 精度测试
```

## 算子描述

- 算子功能：  
strmv算子实现了三角矩阵乘以向量的运算，结果覆盖到输入向量x中。对应的数学表达式为：  
```
x = op(A) * x
```
A为n阶三角矩阵（上三角或下三角），x为n维向量，op(A)可以是A、A的转置或A的共轭转置（实数下同转置）。

矩阵A采用列主序全矩阵存储，需要`lda * n`个元素，其中`lda >= n`。元素`A(i,j)`存储在位置`i + j * lda`，仅对三角区域内的位置有效。

对应的接口为：
```
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
      <td colspan="4" align="center">trmv 参数说明</td>
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
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">trmv</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">lda * N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">核函数名</td><td colspan="6" align="center">arch22: strmv</td></tr>
  <tr><td colspan="6" align="center">arch35: strmv_compute_kernel / strmv_copy_kernel</td></tr>
  </table>

- 算子实现：

  - **arch22**（Atlas A2/A3）：

    采用单kernel实现。Host侧将矩阵按`M0=128`分块，构造128x128的上/下三角掩码矩阵并拷贝至Device。Kernel侧每个AI Core负责一个行分块：将矩阵子块从GM搬运到UB，通过掩码矩阵屏蔽无效三角区域元素，处理单位对角线；将向量x从GM搬运到UB（incx=1时直接拷贝，否则按步长抽取）；通过`vaxpy`/`vmla`+`vcadd`完成矩阵向量乘加；结果写回workspace，最后通过跨核同步和步长写回将结果覆盖到x。

  - **arch35**（Ascend 950）：

    采用两阶段SIMT kernel实现。第一阶段`strmv_compute_kernel`通过SIMT多线程并行，每个线程负责计算输出向量的若干行：根据uplo/trans确定每行的有效列范围，从GM读取矩阵A和向量x的元素，在寄存器中完成乘加累加，将中间结果写入workspace。第二阶段`strmv_copy_kernel`将workspace中的连续结果按incx步长写回x向量。通过模板参数`<UPLO_IS_UPPER, TRANS_IS_N, DIAG_IS_UNIT>`编译期分发8种组合，消除运行时分支。

- 调用实现  
    使用内核调用符<<<>>>调用核函数。arch35下依次调用compute kernel和copy kernel。

## 测试用例覆盖

| 分组 | 用例数 | 覆盖场景 |
|------|--------|----------|
| L0 | 6 | 基础功能：upper/lower × non-transpose/transpose × unit/non-unit、n=0、n=1 |
| L1 | 16 | 共轭转置、非单位步长(incx=2/3)、负步长(incx=-1/-2)、非紧凑lda、转置组合 |
| GEN | 12 | 奇数规模(n=13)、边界规模(n=100)、SIMT规模(n=256/512/1024/2048)、大规模转置 |
| INV | 6 | 非法参数：invalid uplo/trans/diag、incx=0、n<0、lda<n |

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
  bash build.sh --ops=strmv --soc=ascend950 --run
  ```

  其中`--soc` 为**可选**参数，用于指定目标硬件平台（与上文「产品支持情况」对应），决定编译哪个架构的实现：

  | 产品 | `--soc` 取值 | 架构 |
  |------|----------------|:----:|
  | Ascend 950PR / Ascend 950DT | `ascend950` | arch35 |
  | Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | `ascend910_93` | arch22 |
  | Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | `ascend910b` | arch22 |

  执行结果如下，说明精度对比成功。
  ```bash
  [PASS] strmv_test
  ```
