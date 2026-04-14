## Cdot算子实现

## 概述

BLAS Cdot算子实现。

Cdot(Complex Dot Product)算子实现了两个复数向量的点积运算，是BLAS基础线性代数库中的核心算子之一。

该算子支持共轭复数点积运算，广泛应用于信号处理、量子计算和线性代数等领域。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── cdot
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── cdot_test.cpp       // 算子调用样例
```

## 算子描述

- 算子功能：  
cdot算子实现了两个复数向量的点积运算。对应的数学表达式为：  
```
result = x · y = Σ(x[i] * conj(y[i]))  (如果isConj=1)
result = x · y = Σ(x[i] * y[i])        (如果isConj=0)
```
其中x和y是复数向量，conj表示共轭操作。

复数乘法公式：(a + bi) * (c + di) = (ac - bd) + (ad + bc)i

对应的接口为：
```
int aclblasCdot(const float *x, const float *y, float *result,
                const uint32_t n, const uint32_t isConj, void *stream);
```
<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">cdot 参数说明</td>
   </tr>
   <tr>
      <td rowspan="7" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">复数向量的元素个数。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">复数向量，包含 2*n 个float元素（实部和虚部交替存储）。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">复数向量，包含 2*n 个float元素（实部和虚部交替存储）。</td>
   </tr>
   <tr>
      <td align="center">isConj</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">是否对x进行共轭操作。1表示共轭，0表示不共轭。</td>
   </tr>
   <tr>
      <td align="center">result</td>
      <td align="center">device</td>
      <td align="center">out</td>
      <td align="center">复数结果，包含 2 个float元素（实部和虚部）。</td>
   </tr>
</table>


- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Cdot</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">2 * N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">2 * N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">result</td><td align="center">2</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cdot_kernel</td></tr>
  </table>

- 算子实现： 

    1. **数据搬运**：将输入数据从x和y的GM地址分块搬运到UB
    2. **复数分离**：使用`vreducev2`指令将复数的实部和虚部分离
    3. **共轭处理**：如果isConj=1，对x的虚部取反
    4. **复数乘法**：计算(x_real + x_imag*i) * (y_real + y_imag*i)
    5. **累加归约**：使用`cadd_v`指令对所有结果进行累加
    6. **多核归约**：所有核心的结果汇总到core 0，得到最终结果

    **关键优化**：
    - 使用`vreducev2`向量化指令进行复数分离，避免标量循环
    - 使用`cadd_v`向量化指令进行累加归约，提升性能
    - 采用ping-pong流水线，实现数据搬运和计算的重叠
    - 支持多核并行计算，提升大规模数据处理能力

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
  bash build.sh --ops=cdot --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```

## 算子特性

### 性能特性
- **向量化计算**：使用底层向量化指令（vreducev2、cadd_v）实现高性能计算
- **流水线并行**：采用ping-pong双缓冲技术，实现数据搬运和计算的重叠
- **多核并行**：支持多核并行计算，自动分配计算任务

### 数据布局
- **输入向量**：复数以实部和虚部交替存储的方式排列
  - x = [x0_real, x0_imag, x1_real, x1_imag, ..., xn_real, xn_imag]
  - y = [y0_real, y0_imag, y1_real, y1_imag, ..., yn_real, yn_imag]
- **输出结果**：复数结果以实部和虚部连续存储
  - result = [result_real, result_imag]

### 精度要求
- 相对误差 < 1e-5
