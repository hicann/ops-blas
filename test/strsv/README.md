## Strsv算子实现

## 概述

BLAS Strsv算子实现。

Strsv(Triangular packed matrix-vector solve)算子实现了三角矩阵求解运算，是BLAS基础线性代数库中的核心算子之一。

该算子针对三角矩阵的存储特性进行了优化，通过求解三角线性方程组 Ax = b 来计算向量x。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── strsv
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── strsv_test.cpp       // 算子调用样例
```

## 算子描述

- 算子功能：
strsv算子实现了求解三角矩阵方程Ax=b。对应的数学表达式为：
```
A * x = b
```
A为上三角或下三角矩阵，x是待求解的向量。

根据uplo参数，矩阵A的上三角或下三角部分被使用，其他部分被忽略。对角线元素根据diag参数决定是否为单位对角角。

对应的接口为：
```
int aclblasStrsv(aclblasHandle handle,
                 aclblasFillMode uplo,
                 aclblasOperation trans,
                 aclblasDiagType diag,
                 int64_t n,
                 const float *A,
                 int64_t lda,
                 float *x,
                 int64_t incx);
```
<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">strsv 参数说明</td>
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
      <td align="center">aclblas算子流句柄。</td>
   </tr>
   <tr>
      <td align="center">uplo</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">指定使用A的上三角还是下三角部分。</td>
   </tr>
   <tr>
      <td align="center">trans</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">指定是否对A进行转置操作。</td>
   </tr>
   <tr>
      <td align="center">diag</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">指定A的对角线是否为单元对角线。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵A的维度。</td>
   </tr>
   <tr>
      <td align="center">A</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">矩阵A的数据指针，维度为n x lda。</td>
   </tr>
   <tr>
      <td align="center">lda</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵A的leading dimension。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">输入向量b，同时作为输出向量x。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">x中连续元素之间的步长。</td>
   </tr>
</table>


- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">Strsv</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">N x lda</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="6" align="center">strsv_kernel_*</td></tr>
  </table>

- 算子实现：

    将输入数据从A,x的GM地址分块搬运到UB，进行计算后再搬出到x所在的GM地址。

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
  bash build.sh --ops=strsv --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```