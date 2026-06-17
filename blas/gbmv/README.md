## gbmv算子实现

## 概述

BLAS gbmv算子实现。

gbmv(General Banded Matrix-Vector Multiplication)算子实现了带状矩阵与向量的乘法运算，针对带状矩阵的稀疏存储特性进行了优化，支持转置操作和多核并行归约。

## 产品支持情况

| 产品                                                         |  是否支持 |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ✓    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ✓    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    ✓    |

## 目录结构介绍

```
blas/gbmv/
├── README.md                   // 说明文档
└── arch35/
    ├── sgbmv_host.cpp          // Host 侧实现
    ├── sgbmv_kernel.cpp        // Kernel 侧实现
    └── sgbmv_tiling_data.h     // Tiling 数据结构
```

测试代码位于 `test/gbmv/`：

```
test/gbmv/
├── CMakeLists.txt              // 编译工程文件
├── sgbmv_param.h               // 参数结构体（继承 BlasTestParamBase）
├── sgbmv_golden.h              // CPU golden（签名与 BLAS API 一致）
└── arch35/
    ├── sgbmv_npu_wrapper.h     // NPU wrapper（封装 aclrtMalloc/H2D/kernel/D2H/free）
    ├── sgbmv_test.cpp          // 精度测试（GTest 入口）
    └── sgbmv_test.csv          // 精度测试用例表
```

## 算子描述

- 算子功能：  
gbmv算子实现了带状矩阵乘以向量的运算。对应的数学表达式为：  
```
y = alpha * op(A) * x + beta * y
```
A为带状矩阵，x和y是向量，alpha和beta是标量，op(A)可以是A或A的转置。

带状矩阵A采用LAPACK带状存储格式，需要`lda * n`个元素存储，其中`lda >= kl + ku + 1`。元素`A(i,j)`存储在位置`ku + i - j + j * lda`，仅对满足`max(0, j-ku) <= i <= min(m-1, j+kl)`的位置有效。

对应的接口为：
```
aclblasStatus_t aclblasSgbmv(
    aclblasHandle_t handle,
    aclblasOperation_t trans,
    int64_t m,
    int64_t n,
    int64_t kl,
    int64_t ku,
    const float *alpha,
    const float *A,
    int64_t lda,
    const float *x,
    int64_t incx,
    const float *beta,
    float *y,
    int64_t incy);
```
<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">gbmv 参数说明</td>
   </tr>
   <tr>
      <td rowspan="14" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">handle</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">aclbLAS 库上下文句柄。</td>
   </tr>
   <tr>
      <td align="center">trans</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵操作类型：ACLBLAS_OP_N(不转置)、ACLBLAS_OP_T(转置)、ACLBLAS_OP_C(共轭转置，实数下同T)。</td>
   </tr>
   <tr>
      <td align="center">m</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵 A 的行数。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵 A 的列数。</td>
   </tr>
   <tr>
      <td align="center">kl</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">A 的下带宽（主对角线以下的非零对角线数）。</td>
   </tr>
   <tr>
      <td align="center">ku</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">A 的上带宽（主对角线以上的非零对角线数）。</td>
   </tr>
   <tr>
      <td align="center">alpha</td>
      <td align="center">host/device</td>
      <td align="center">in</td>
      <td align="center">用于乘法的 &lt;type&gt; 标量。</td>
   </tr>
   <tr>
      <td align="center">A</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">带状矩阵 &lt;type&gt; 数组，维度为 lda x n。</td>
   </tr>
   <tr>
      <td align="center">lda</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">带状矩阵 A 存储的主维长度，lda >= kl + ku + 1。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">&lt;type&gt; 向量，trans='N'时包含n个元素，否则包含m个元素。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">x 中连续元素之间的步长。</td>
   </tr>
   <tr>
      <td align="center">beta</td>
      <td align="center">host/device</td>
      <td align="center">in</td>
      <td align="center">用于乘法的 &lt;type&gt; 标量。如果 beta == 0，则 y 不必是有效输入。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">&lt;type&gt; 向量，trans='N'时包含m个元素，否则包含n个元素。</td>
   </tr>
   <tr>
      <td align="center">incy</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">y 中连续元素之间的步长。</td>
   </tr>
</table>


- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">gbmv</td></tr>
  </tr>
  <tr><td rowspan="6" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">lda * N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">x</td><td align="center">X_COUNT</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">Y_COUNT</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">alpha</td><td align="center">1</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">beta</td><td align="center">1</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">Y_COUNT</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="6" align="center">gbmv_kernel</td></tr>
  </table>

- 算子实现： 

    输入数据通过Host侧GatherStrided将带步长的向量收集为连续内存，经H2D拷贝到Device。Kernel将矩阵列段和向量数据从GM分块搬运到UB，完成乘加计算后，通过SetAtomicAdd多核并行归约到输出向量。结果通过ScatterStrided按输出步长写回Host。

- 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 测试用例覆盖

| 分组 | 用例数 | 覆盖场景 |
|------|--------|----------|
| L0 trans=N | 7 | 基础功能、alpha/beta边界值、非单位步长、矩形矩阵 |
| L0 trans=T/C | 7 | 转置基础功能、beta=0、矩形、共轭转置 |
| L1 trans=N | 19 | 带宽变体、边界形状、负步长、非紧凑lda、大规模、全零输出 |
| L1 trans=T/C | 18 | 转置带宽变体、负步长、大规模矩形、全零输出 |

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
  bash build.sh --ops=gbmv --soc=ascend950 --run
  ```

  其中`--soc` 为**可选**参数，用于指定目标硬件平台（与上文「产品支持情况」对应）。按实际硬件选用：

  | 产品 | `--soc` 取值 |
  |------|----------------|
  | Ascend 950PR / Ascend 950DT | `ascend950` |
  | Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | `ascend910_93` |
  | Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | `ascend910b` |

  执行结果如下，说明精度对比成功。
  ```bash
  [PASS] gbmv_test
  ```
