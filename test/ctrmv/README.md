## Ctrmv算子实现

## 概述

BLAS Ctrmv算子实现。

Ctrmv(复数三角矩阵-向量乘法)算子实现了复数三角矩阵与向量的乘法运算，是BLAS基础线性代数库中的Level 2算子之一。

该算子计算 x = op(A) * x，其中A为复数三角矩阵，x为复数向量，op(A)由trans参数决定。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── ctrmv
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── ctrmv_test.cpp      // 算子调用样例
```

## 算子描述

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

对应的接口为：
```
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
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">n × n</td><td align="center">complex&lt;float&gt;</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">x</td><td align="center">n</td><td align="center">complex&lt;float&gt;</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">ctrmv_kernel</td></tr>
  </table>

- 算子实现： 

    将输入数据从A和x的GM地址分块搬运到UB，进行复数三角矩阵-向量乘法计算后再将结果从workspace回写到x所在的GM地址。支持上三角和下三角模式，支持不转置、转置和共轭转置操作。

- 调用实现  
    使用内核调用符<<<>>>调用核函数。 

## 约束说明

- n的取值范围为[1, 8192]
- 仅支持complex&lt;float&gt;数据类型
- incx > 0
- lda > 0

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
  bash build.sh --ops=ctrmv --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```
