## Cgemm算子实现

## 概述

BLAS Cgemm算子实现。

Cgemm(Complex General Matrix-Matrix multiplication)算子实现了复数矩阵乘法运算，是BLAS基础线性代数库中的核心算子之一。

该算子实现复数矩阵乘法：`C = alpha * op(A) * op(B) + beta * C`（支持转置和共轭转置）

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── cgemm
│   ├── cgemm_kernel.cpp    // 算子核函数实现（AIC+AIV混编）
│   └── cgemm_host.cpp      // 算子Host端实现
├── test/cgemm
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── cgemm_test.cpp      // 算子调用样例
```

## 算子描述

- 算子功能：  
cgemm算子实现了复数矩阵A与矩阵B的乘法运算，并加到矩阵C上。对应的数学表达式为：
```
C = alpha * op(A) * op(B) + beta * C
```
其中op(A)可以是：
- A（不转置，transA=N）
- A^T（转置，transA=T）
- A^H（共轭转置，transA=C）

op(B)同理。

复数乘法公式：`(a+bi) * (c+di) = (ac-bd) + (ad+bc)i`

- 对应的接口：
```cpp
int aclblasCgemm(aclblasHandle handle,
                 aclblasOperation transA,
                 aclblasOperation transB,
                 const int64_t m, const int64_t n, const int64_t k,
                 const std::complex<float> &alpha,
                 const std::complex<float> *A, const int64_t lda,
                 const std::complex<float> *B, const int64_t ldb,
                 const std::complex<float> &beta,
                 std::complex<float> *C, const int64_t ldc,
                 void *stream);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">cgemm 参数说明</td>
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
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">ACL流handle，用于传入stream。</td>
   </tr>
   <tr>
      <td align="center">transA</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">矩阵A操作类型：N=不转置，T=转置，C=共轭转置。</td>
   </tr>
   <tr>
      <td align="center">transB</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">矩阵B操作类型：N=不转置，T=转置，C=共轭转置。</td>
   </tr>
   <tr>
      <td align="center">m</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵A的行数、矩阵C的行数。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵B的列数、矩阵C的列数。</td>
   </tr>
   <tr>
      <td align="center">k</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵A的列数、矩阵B的行数。</td>
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
      <td align="center">复数矩阵A。</td>
   </tr>
   <tr>
      <td align="center">lda</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵A的主维长度。</td>
   </tr>
   <tr>
      <td align="center">B</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">复数矩阵B。</td>
   </tr>
   <tr>
      <td align="center">ldb</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵B的主维长度。</td>
   </tr>
   <tr>
      <td align="center">beta</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">复数标量beta。</td>
   </tr>
   <tr>
      <td align="center">C</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">复数矩阵C。</td>
   </tr>
   <tr>
      <td align="center">ldc</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵C的主维长度。</td>
   </tr>
</table>

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Cgemm</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">M * K</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  <tr><td align="center">B</td><td align="center">K * N</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">C</td><td align="center">M * N</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cgemm</td></tr>
  </table>

- 算子实现： 

    本算子为AIC+AIV混编算子，采用流水线并行设计：
    - AIV端（Vector Core）：执行ascblasCgemmPre进行虚实分离，将复数矩阵拆分为实部矩阵和虚部矩阵
    - AIC端（Cube Core）：执行4次实矩阵乘法ascblasSmatmul，计算结果的实部和虚部
    - AIV端（Vector Core）：执行ascblasCgemmFinal进行虚实合并，组装最终复数结果
    
    同步机制：
    - 使用FftsCrossCoreSync进行AIC和AIV之间的同步
    - 使用WaitFlagDev等待跨核事件
    
    使用vreducev2进行虚实分离，使用Gather进行虚实合并。

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
  bash build.sh --ops=cgemm --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  Testing cgemm:
  Output: ...
  Golden: ...
  [Success] Case accuracy is verification passed.
  [PASS] cgemm_test
  ```