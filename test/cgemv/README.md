## Cgemv算子实现

## 概述

BLAS Cgemv算子实现。

Cgemv(Complex General Matrix-Vector multiplication)算子实现了复数矩阵与向量乘法运算，是BLAS基础线性代数库中的核心算子之一。

该算子实现复数矩阵向量乘法：`y = alpha * A * x + beta * y`（支持转置和共轭转置）

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── cgemv
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── cgemv_test.cpp       // 算子调用样例
```

## 算子描述

- 算子功能：  
cgemv算子实现了复数矩阵A与向量x的乘法运算，并加到向量y上。对应的数学表达式为：
```
y = alpha * op(A) * x + beta * y
```
其中op(A)可以是：
- A（不转置，trans=N）
- A^T（转置，trans=T）
- A^H（共轭转置，trans=C）

复数乘法公式：`(a+bi) * (c+di) = (ac-bd) + (ad+bc)i`

- 对应的接口：
```cpp
int aclblasCgemv(aclblasHandle handle,
                  aclblasOperation trans,
                  const int64_t m, const int64_t n,
                  const std::complex<float> &alpha,
                  const std::complex<float> *A, const int64_t lda,
                  const std::complex<float> *x, const int64_t incx,
                  const std::complex<float> &beta,
                  std::complex<float> *y, const int64_t incy,
                  void *stream);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">cgemv 参数说明</td>
   </tr>
   <tr>
      <td rowspan="12" align="center">参数列表</td>
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
      <td align="center">trans</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">矩阵操作类型：N=不转置，T=转置，C=共轭转置。</td>
   </tr>
   <tr>
      <td align="center">m</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵A的行数。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵A的列数。</td>
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
      <td align="center">m x n复数矩阵。</td>
   </tr>
   <tr>
      <td align="center">lda</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵A的主维长度。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">向量x（长度取决于trans）。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">x中连续元素之间的步长。</td>
   </tr>
   <tr>
      <td align="center">beta</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">复数标量beta。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">向量y（长度取决于trans）。</td>
   </tr>
   <tr>
      <td align="center">incy</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">y中连续元素之间的步长。</td>
   </tr>
</table>

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Cgemv</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">512 * 256</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  <tr><td align="center">x</td><td align="center">256</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">512</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cgemv_no_trans / cgemv_do_trans</td></tr>
  </table>

- 算子实现： 

    根据是否转置，选择不同的kernel函数：
    - cgemv_no_trans：不转置情况，矩阵按列分块，每列与向量元素相乘后累加到y
    - cgemv_do_trans：转置情况，矩阵按行分块，每行与向量元素相乘后累加到y
    
    使用vreducev2进行虚实分离，使用vgather进行虚实合并，利用AsdopsBuffer进行乒乓缓冲优化。

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
  bash build.sh --ops=cgemv --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  Testing cgemv (trans=N):
  [Success] Case accuracy is verification passed.
  Testing cgemv (trans=T):
  [Success] Case accuracy is verification passed.
  [PASS] cgemv_test
  ```