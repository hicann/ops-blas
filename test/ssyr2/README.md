## Ssyr2算子实现

## 概述

BLAS Ssyr2算子实现。

Ssyr2(Symmetric Rank-2 Update)算子实现了单精度向量的外积并将结果加到一个矩阵上，是BLAS基础线性代数库中的核心算子之一。

该算子实现对称秩2更新：`A = alpha * x * y^T + alpha * y * x^T + A`

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── ssyr2
│   ├── ssyr2_kernel.cpp    // 算子核函数实现
│   └── ssyr2_host.cpp      // 算子Host端实现
├── test/ssyr2
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── ssyr2_test.cpp      // 算子调用样例
```

## 算子描述

- 算子功能：  
ssyr2算子实现了单精度向量的外积并将结果加到一个矩阵上。对应的数学表达式为：
```
A = alpha * x * y^T + alpha * y * x^T + A
```

- 对应的接口：
```cpp
aclblasSsyr2(aclblasHandle handle,
             aclblasFillMode uplo,
             const int64_t n,
             const float alpha,
             uint8_t *x,
             const int64_t incx,
             uint8_t *y,
             const int64_t incy,
             uint8_t *A,
             const int64_t lda)
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">ssyr2 参数说明</td>
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
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">算子的句柄。</td>
   </tr>
   <tr>
      <td align="center">uplo</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">指定矩阵A的存储格式。ASDBLAS_FILL_MODE_LOWER:下三角，ASDBLAS_FILL_MODE_UPPER:上三角。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">向量x和y中的元素个数，矩阵A的行列数。取值范围[1, 8192]。</td>
   </tr>
   <tr>
      <td align="center">alpha</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">标量alpha，向量乘积缩放因子。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">输入向量，对应公式中的'x'。数据类型支持FLOAT32，数据格式支持ND，shape为[n]。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">x相邻元素间的内存地址偏移量（当前约束为1）。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">输入向量，对应公式中的'y'。数据类型支持FLOAT32，数据格式支持ND，shape为[n]。</td>
   </tr>
   <tr>
      <td align="center">incy</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">y相邻元素间的内存地址偏移量（当前约束为1）。</td>
   </tr>
   <tr>
      <td align="center">A</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">输入/输出矩阵，对应公式中的'A'。数据类型支持FLOAT32，数据格式支持ND，shape为[n, n]。</td>
   </tr>
   <tr>
      <td align="center">lda</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵A的每列元素的存储步长（当前约束为n）。</td>
   </tr>
</table>

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Ssyr2</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[n]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[n]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">A</td><td align="center">[n, n]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">ssyr2</td></tr>
  </table>

- 约束说明：
  - 输入的元素个数n当前覆盖支持[1, 8192]。
  - 算子输入shape为[n]、[n]、[n, n]，输出shape为[n, n]。
  - 算子实际计算时，不支持ND高维度运算（不支持维度≥3的运算）。

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
  bash build.sh --ops=ssyr2 --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  Testing ssyr2:
  Output: ...
  Golden: ...
  [Success] Case accuracy is verification passed.
  [PASS] ssyr2_test
  ```