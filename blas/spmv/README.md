## spmv算子实现

## 概述

BLAS spmv算子实现。

Spmv(Symmetric Packed Matrix-Vector Multiplication)算子实现了对称压缩矩阵与向量的乘法运算，是BLAS基础线性代数库中的核心算子之一。

该算子针对对称矩阵的存储特性进行了优化，采用压缩存储格式以节省内存空间，并高效完成矩阵与向量的乘加运算。

## 产品支持情况

| 产品                                                         |  是否支持 | 架构 |
| :----------------------------------------------------------- |:-------:|:----:|
| Ascend 950PR / Ascend 950DT                                  |    ✓    | arch35 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品                    |    ✓    | arch22 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品                    |    ✓    | arch22 |

## 目录结构介绍

```
blas/spmv/
├── README.md                       // 说明文档
├── arch22/
│   ├── sspmv_host.cpp              // Host 侧实现（arch22）
│   └── sspmv_kernel.cpp            // Kernel 侧实现（arch22）
└── arch35/
    ├── sspmv_host.cpp              // Host 侧实现（arch35）
    ├── sspmv_kernel.cpp            // Kernel 侧实现（arch35）
    └── sspmv_tiling_data.h         // Tiling 数据结构（arch35）
```

## 算子描述

### arch22 接口

- 算子功能：

spmv算子实现了将对称压缩矩阵乘以向量。对应的数学表达式为：
```
z = alpha * A * x + beta * y
```
A为对称压缩矩阵，x和y是向量，alpha和beta是标量

对称矩阵A的下三角部分元素按行连续打包储存，元素`A(i,j)`储存在位置`AP[j + i * (i + 1) / 2]`中，且`i >= j`，对应的对称部分通过已有元素推断得出。压缩对称矩阵格式仅需要`n * (n + 1) / 2`个元素储存。

对应的接口为：
```cpp
int aclblasSpmv(const float *aPacked, const float *x, const float *y, float *z,
                const float alpha, const float beta,
                const int64_t n, const int64_t incx, const int64_t incy, void *stream);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">spmv 参数说明（arch22）</td>
   </tr>
   <tr>
      <td rowspan="10" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">对称压缩矩阵 A 的行数和列数。</td>
   </tr>
   <tr>
      <td align="center">alpha</td>
      <td align="center">host/device</td>
      <td align="center">in</td>
      <td align="center">标量乘数。</td>
   </tr>
   <tr>
      <td align="center">aPacked</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">对称压缩矩阵，n*(n+1)/2 个元素。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">输入向量，包含 n 个元素。</td>
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
      <td align="center">标量乘数。如果 beta == 0，则 y 不必是有效输入。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">输入向量，包含 n 个元素。</td>
   </tr>
   <tr>
      <td align="center">incy</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">y 中连续元素之间的步长。</td>
   </tr>
   <tr>
      <td align="center">z</td>
      <td align="center">device</td>
      <td align="center">out</td>
      <td align="center">输出向量，包含 n 个元素。</td>
   </tr>
</table>

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Spmv</td></tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A (packed)</td><td align="center">n*(n+1)/2</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">spmv_kernel</td></tr>
  </table>

- 算子实现：

    将输入数据从A,x,y的GM地址分块搬运到UB，进行计算后再搬出到z所在的GM地址。

### arch35 接口

- 算子功能：

sspmv 实现了对称压缩矩阵与向量的乘法运算：
```
y = alpha * A * x + beta * y
```
其中 A 为 n×n 对称矩阵，以 packed format 压缩存储，仅存上三角或下三角部分（共 n(n+1)/2 个元素），对称部分通过已有元素推断。

对应的接口为：
```cpp
aclblasStatus_t aclblasSspmv(aclblasHandle handle,
                 aclblasFillMode uplo, int n, const float *alpha,
                 const float *AP, const float *x, int incx, const float *beta,
                 float *y, int incy);
```

| 参数 | in/out | 设备 | 含义 |
|------|--------|------|------|
| handle | in | host | aclblas 库句柄，携带 stream |
| uplo | in | host | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122) |
| n | in | host | 方阵阶数，n >= 0 |
| alpha | in | host | 标量 alpha 的指针 |
| AP | in | device | 对称压缩矩阵，共 n(n+1)/2 个元素 |
| x | in | device | 输入向量，n 个元素 |
| incx | in | host | x 的步长，incx != 0（可正可负） |
| beta | in | host | 标量 beta 的指针 |
| y | in/out | device | 输入/输出向量，n 个元素 |
| incy | in | host | y 的步长，incy != 0（可正可负） |

**注意**：AP、x、y 必须为 device 侧指针。alpha、beta 为 host 侧指针。stream 通过 `aclblasSetStream(handle, stream)` 绑定到 handle。

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Sspmv</td></tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">AP</td><td align="center">n*(n+1)/2</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sspmv_kernel</td></tr>
  </table>

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
  bash build.sh --ops=spmv --run
  bash build.sh --ops=spmv --soc=ascend950 --run
  ```

  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```
