## ComplexMatDot算子实现

## 概述

BLAS ComplexMatDot算子实现。

ComplexMatDot(复数矩阵点乘)算子实现了两个复数矩阵的逐元素乘法运算，是BLAS基础线性代数库中的扩展算子之一。

该算子针对复数运算特性进行了优化，使用GatherMask操作高效完成复数矩阵的逐元素乘法。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── complex_mat_dot
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── complex_mat_dot_test.cpp // 算子调用样例
```

## 算子描述

- 算子功能：  
ComplexMatDot算子实现了两个复数矩阵的逐元素乘法。对应的数学表达式为：  
```
result[i, j] = matx[i, j] * maty[i, j]
```
matx和maty是复数矩阵，result是输出复数矩阵

复数乘法公式：(a + bi) * (c + di) = (ac - bd) + (ad + bc)i

对应的接口为：
```
int aclblasComplexMatDot(const float *matx, const float *maty, float *result,
                         const int64_t m, const int64_t n, void *stream);
```
<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">complex_mat_dot 参数说明</td>
   </tr>
   <tr>
      <td rowspan="6" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">m</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵的行数。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵的列数。</td>
   </tr>
   <tr>
      <td align="center">matx</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">复数矩阵，维度为 m × n，存储为 2*m*n 个float。</td>
   </tr>
   <tr>
      <td align="center">maty</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">复数矩阵，维度为 m × n，存储为 2*m*n 个float。</td>
   </tr>
   <tr>
      <td align="center">result</td>
      <td align="center">device</td>
      <td align="center">out</td>
      <td align="center">复数矩阵，维度为 m × n，存储为 2*m*n 个float。</td>
   </tr>
</table>


- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">ComplexMatDot</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">matx</td><td align="center">m × n</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  <tr><td align="center">maty</td><td align="center">m × n</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">result</td><td align="center">m × n</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">complex_mat_dot_kernel</td></tr>
  </table>

- 算子实现： 

    将输入数据从matx和maty的GM地址分块搬运到UB，使用GatherMask分离实部和虚部，进行复数乘法计算后再搬出到result所在的GM地址。

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
  bash build.sh --ops=complex_mat_dot --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```
