## CgemmBatched算子实现

## 概述

BLAS CgemmBatched算子实现。

CgemmBatched(Complex General Matrix-Matrix multiplication Batched)算子实现了批量复数矩阵乘法运算，是BLAS基础线性代数库中的核心算子之一。

该算子实现批量复数矩阵乘法：`C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]`

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── cgemm_batched
│   ├── cgemm_batched_kernel.cpp    // 算子核函数实现
│   └── cgemm_batched_host.cpp      // 算子Host端实现
├── test/cgemm_batched
│   ├── CMakeLists.txt              // 编译工程文件
│   ├── README.md                   // 说明文档
│   └── cgemm_batched_test.cpp      // 算子调用样例
```

## 算子描述

- 算子功能：  
cgemm_batched算子实现了批量复数矩阵A与矩阵B的乘法运算，并加到矩阵C上。对应的数学表达式为：
```
C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
```

- 对应的接口：
```cpp
aclblasCgemmBatched(aclblasHandle handle,
                     aclblasOperation transa,
                     aclblasOperation transb,
                     const int64_t m,
                     const int64_t n,
                     const int64_t k,
                     const std::complex<float> &alpha,
                     uint8_t *A,
                     const int64_t lda,
                     uint8_t *B,
                     const int64_t ldb,
                     const std::complex<float> &beta,
                     uint8_t *C,
                     const int64_t ldc,
                     const int64_t batchCount)
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">cgemm_batched 参数说明</td>
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
      <td align="center">算子的句柄。</td>
   </tr>
   <tr>
      <td align="center">transa</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">指定矩阵A是否需要转置，取值必须为N（不转置）。</td>
   </tr>
   <tr>
      <td align="center">transb</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">指定矩阵B是否需要转置，取值必须为N（不转置）。</td>
   </tr>
   <tr>
      <td align="center">m</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵C的行数，取值范围为：{1-32}。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵C的列数，取值范围为：{1-32}。</td>
   </tr>
   <tr>
      <td align="center">k</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵A和B的公共维度，取值范围为：{1-32}。</td>
   </tr>
   <tr>
      <td align="center">alpha</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">复数标量alpha，用于乘以矩阵乘法的结果，取值必须为1+0j。</td>
   </tr>
   <tr>
      <td align="center">A</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">输入的矩阵，对应公式中的'A'。数据类型支持COMPLEX64，数据格式支持ND，shape为[batchCount, m, k]。</td>
   </tr>
   <tr>
      <td align="center">lda</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">A左右相邻元素间的内存地址偏移量，取值和k相等。</td>
   </tr>
   <tr>
      <td align="center">B</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">输入的矩阵，对应公式中的'B'。数据类型支持COMPLEX64，数据格式支持ND，shape为[batchCount, k, n]。</td>
   </tr>
   <tr>
      <td align="center">ldb</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">B左右相邻元素间的内存地址偏移量，取值和n相等。</td>
   </tr>
   <tr>
      <td align="center">beta</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">复数标量beta，用于乘以矩阵C。取值必须为0+0j。</td>
   </tr>
   <tr>
      <td align="center">C</td>
      <td align="center">device</td>
      <td align="center">out</td>
      <td align="center">对应公式中的'C'。数据类型支持COMPLEX64，数据格式支持ND，shape为[batchCount, m, n]。</td>
   </tr>
   <tr>
      <td align="center">ldc</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">C左右相邻元素间的内存地址偏移量，取值和n相等。</td>
   </tr>
   <tr>
      <td align="center">batchCount</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">批次数量。取值范围为{12 - 26208}。</td>
   </tr>
</table>

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">CgemmBatched</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">[batchCount, M, K]</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  <tr><td align="center">B</td><td align="center">[batchCount, K, N]</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">C</td><td align="center">[batchCount, M, N]</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cgemm_batched</td></tr>
  </table>

- 约束说明：
  - 算子实际计算时，只支持3维ND运算。
  - 算子输入数据为行主序，输入shape为[batchCount, m, k]、[batchCount, k, n]、[batchCount, m, n]，输出shape为[batchCount, m, n]。

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
  bash build.sh --ops=cgemm_batched --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  Testing cgemm_batched:
  Output: ...
  Golden: ...
  [Success] Case accuracy is verification passed.
  [PASS] cgemm_batched_test
  ```