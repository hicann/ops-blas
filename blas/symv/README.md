## symv算子实现

## 概述

BLAS symv算子实现。

symv(Symmetric Matrix-Vector Multiplication)算子实现了对称矩阵与向量的乘法运算，是BLAS基础线性代数库中的核心算子之一。

该算子针对对称矩阵的存储特性进行了优化，并高效完成矩阵与向量的乘加运算。

## 产品支持情况

| 产品                                                         |  是否支持 | 架构 |
| :----------------------------------------------------------- |:-------:|:----:|
| Ascend 950PR / Ascend 950DT                                  |    ✓    | arch35 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品                    |    ✓    | arch22 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品                    |    ✓    | arch22 |

## 目录结构介绍

```
blas/symv/
├── README.md                       // 说明文档
├── arch22/
│   ├── ssymv_host.cpp              // Host 侧实现（arch22）
│   └── ssymv_kernel.cpp            // Kernel 侧实现（arch22）
└── arch35/
    ├── ssymv_host.cpp              // Host 侧实现（arch35）
    ├── ssymv_kernel.cpp            // Kernel 侧实现（arch35）
    └── ssymv_tiling_data.h         // Tiling 数据结构（arch35）

test/symv/ssymv/
├── CMakeLists.txt                  // 测试构建文件
├── ssymv_param.h                   // CSV 参数解析结构体
├── ssymv_golden.h                  // CPU golden（cblas_ssymv）
├── arch22/
│   └── ssymv_test.cpp              // 测试代码（arch22）
└── arch35/
    ├── ssymv_test.cpp              // 测试代码（arch35，GTest 参数化）
    ├── ssymv_test.csv              // CSV 测试用例
    └── ssymv_npu_wrapper.h         // NPU wrapper
```

## 算子描述

- 算子功能：  
symv算子实现了对称矩阵乘以向量。对应的数学表达式为：  
```
y = alpha * A * x + beta * y
```
A为对称矩阵，x和y是向量，alpha和beta是标量。结果直接写回y（in-place）。

对称矩阵A仅存储上三角或下三角部分（由uplo参数指定），未存储的部分通过对称性推断。矩阵按列主序（column-major）存储，主维为lda。

对应的接口为：
```
aclblasStatus_t aclblasSsymv(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float *alpha,
             const float *A, int lda, const float *x, int incx,
             const float *beta, float *y, int incy);
```
<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">symv 参数说明</td>
   </tr>
   <tr>
      <td rowspan="11" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">uplo</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">指定矩阵 A 存储上三角（ACLBLAS_UPPER）或下三角（ACLBLAS_LOWER）。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">对称矩阵 A 的行数和列数。</td>
   </tr>
   <tr>
      <td align="center">alpha</td>
      <td align="center">host/device</td>
      <td align="center">in</td>
      <td align="center">用于乘法的 float 标量。</td>
   </tr>
   <tr>
      <td align="center">A</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">对称矩阵 float 数组，维度为 lda x n。仅存储 uplo 指定的三角部分。</td>
   </tr>
   <tr>
      <td align="center">lda</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">用于存储矩阵A的二维数组的主维，lda >= max(1, n)。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">float 向量，包含 n 个元素。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">x 中连续元素之间的步长，incx != 0。</td>
   </tr>
   <tr>
      <td align="center">beta</td>
      <td align="center">host/device</td>
      <td align="center">in</td>
      <td align="center">用于乘法的 float 标量。如果 beta == 0，则 y 不必是有效输入。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">float 向量，包含 n 个元素。输入为初始 y 值，输出为计算结果。</td>
   </tr>
   <tr>
      <td align="center">incy</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">y 中连续元素之间的步长，incy != 0。</td>
   </tr>
</table>


- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">symv</td></tr>
  </tr>
  <tr><td rowspan="6" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">lda x N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">alpha</td><td align="center">1</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">beta</td><td align="center">1</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="6" align="center">ssymv_kernel (arch35) / symv_kernel (arch22)</td></tr>
  </table>

- 算子实现： 

    **arch35 (SIMT)**：采用 SIMT 编程模型，使用 grid-stride loop 实现线程级并行。每个线程计算输出向量 y 的一个或多个元素，通过 `asc_vf_call` 分发到不同模板特化（按 uplo 区分 UPPER/LOWER）。Tiling 数据通过传值方式（by value）从 host 传入 kernel，无需分配 GM 设备内存。Host 侧通过 `GetAivCoreCount()` 获取 AIV 核数，异步 launch kernel 后直接返回。

    **arch22 (AscendC)**：采用 AscendC 编程模型，使用 tile-based 分块计算。将输入数据从 A、x、y 的 GM 地址分块搬运到 UB，进行计算后再搬出到 z（临时缓冲区）所在的 GM 地址，最后拷贝回 y。Tiling 数据通过 GM 地址传入 kernel（结构体较大，不适合传值）。

- 调用实现  
    使用内核调用符<<<>>>调用核函数。 

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
  bash build.sh --ops=symv --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明所有测试用例通过。
  ```bash
  [  PASSED  ] 32 tests.
  ```