## tpmv算子实现

## 概述

BLAS tpmv算子实现。

tpmv(Triangular Packed Matrix-Vector Multiplication)算子实现了三角矩阵与向量的乘法运算，是BLAS基础线性代数库中的核心算子之一。

该算子针对对称矩阵的存储特性进行了优化，采用压缩存储格式以节省内存空间，并高效完成矩阵与向量的乘加运算。

## 支持的产品

- Ascend 950（Atlas A5 训练/推理系列）
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
blas/tpmv/
├── README.md                       // 说明文档
├── arch22/
│   ├── stpmv_host.cpp              // Host 侧实现（arch22）
│   └── stpmv_kernel.cpp            // Kernel 侧实现（arch22）
└── arch35/
    ├── stpmv_host.cpp              // Host 侧实现（arch35）
    ├── stpmv_kernel_simt.cpp       // Kernel 侧实现（arch35 SIMT）
    └── stpmv_tiling_data.h         // Tiling 数据结构（arch35）
```

## 算子描述

- 算子功能：  
tpmv算子实现了将三角矩阵乘以向量（in-place）。对应的数学表达式为：  
```
x = A * x
```
A为三角压缩矩阵，x是向量（既是输入也是输出）

三角矩阵A的元素按列连续打包储存（BLAS 标准 column-major packed 格式），压缩三角矩阵格式仅需要`n * (n + 1) / 2`个元素储存。
- 下三角：元素`A(i,j)`储存在位置`AP[i + (2n-j-1)*j/2]`中，且`i >= j`。
- 上三角：元素`A(i,j)`储存在位置`AP[i + j*(j+1)/2]`中，且`i <= j`。

**arch35 接口（Ascend 950）：**

```
aclblasStatus_t aclblasStpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int n, const float *AP,
    float *x, int incx);
```

**arch22 接口（Atlas A2/A3）：**

```
aclblasStatus_t aclblasStpmv_legacy(
    aclblasHandle_t handle, aclblasFillMode uplo, aclblasOperation trans,
    aclblasDiagType diag, int64_t n, const float *aPacked,
    const float *x, float *y, int64_t incx);
```

> `aclblasStpmv_legacy` 为早期 arch22 贡献接口，该接口后续可能调整或删除，建议新代码使用标准 `aclblasStpmv` 接口。
<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">tpmv 参数说明</td>
   </tr>
   <tr>
      <td rowspan="12" align="center">参数列表</td>
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
      <td align="center">AP</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">对称压缩矩阵 &lt;type&gt; 数组，维度为 n x n。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">&lt;type&gt; 向量，包含 n 个元素。既是输入也是输出（in-place）。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">x 中连续元素之间的步长。</td>
   </tr>
</table>


- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">tpmv</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">N * (N + 1) /2</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">x (in-place)</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="6" align="center">stpmv_simt_kernel / stpmv_scatter_kernel</td></tr>
  </table>

- 算子实现： 

    将输入数据从 A、x 的 GM 地址搬运到 device，kernel 读取 x 并写入临时输出 buffer，计算完成后将结果拷贝回 x（in-place 语义）。

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
  bash build.sh --ops=tpmv --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```