## Sdot算子实现

## 概述

BLAS Sdot算子实现。

Sdot(Real Dot Product)算子实现了两个实数向量的点积运算，是BLAS基础线性代数库中的核心算子之一。

该算子广泛应用于信号处理、统计学和线性代数等领域。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── sdot
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── sdot_test.cpp       // 算子调用样例
```

## 算子描述

- 算子功能：  
sdot算子实现了两个实数向量的点积运算。对应的数学表达式为：  
```
result = x · y = Σ(x[i] * y[i])  for i = 0 to n-1
```
其中x和y是实数向量。

对应的接口为：
```
int aclblasSdot(aclblasHandle handle, const float *x, const float *y, float *result,
                const int64_t n, const int64_t incx, const int64_t incy);
```
<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">sdot 参数说明</td>
   </tr>
   <tr>
      <td rowspan="6" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">handle</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">ACL BLAS句柄，用于传入stream。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">实数向量的元素个数。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">实数向量，包含 n 个float元素。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">实数向量，包含 n 个float元素。</td>
   </tr>
   <tr>
      <td align="center">result</td>
      <td align="center">device</td>
      <td align="center">out</td>
      <td align="center">实数结果，包含 1 个float元素。</td>
   </tr>
</table>


- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Sdot</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">result</td><td align="center">1</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sdot_kernel</td></tr>
  </table>

- 算子实现： 

    1. **数据搬运**：将输入数据从x和y的GM地址分块搬运到UB
    2. **乘法运算**：使用`mul_v`指令计算x[i] * y[i]
    3. **累加归约**：使用`cadd_v`指令对所有乘法结果进行累加
    4. **多核归约**：所有核心的结果汇总到core 0，得到最终结果

    **关键优化**：
    - 使用`mul_v`和`cadd_v`向量化指令进行乘法和累加，提升性能
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
  bash build.sh --ops=sdot --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```

## 算子特性

### 性能特性
- **向量化计算**：使用底层向量化指令（mul_v、cadd_v）实现高性能计算
- **流水线并行**：采用ping-pong双缓冲技术，实现数据搬运和计算的重叠
- **多核并行**：支持多核并行计算，自动分配计算任务

### 数据布局
- **输入向量**：实数向量连续存储
  - x = [x0, x1, x2, ..., xn-1]
  - y = [y0, y1, y2, ..., yn-1]
- **输出结果**：实数结果
  - result = Σ(x[i] * y[i])

### 精度要求
- 相对误差 < 1e-5