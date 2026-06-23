## Dot算子实现

## 概述

BLAS Dot算子实现，包含实数点积（Sdot）和复数点积（Cdot）两类接口。

Dot（点积）算子实现了两个向量的点积运算，是BLAS基础线性代数库中的核心算子之一。

该算子包含以下接口：
- **aclblasSdot**：实数向量点积，result = Σ(x[i] * y[i])
- **aclblasCdotu**：无共轭的复数点积，result = Σ(x[i] * y[i])
- **aclblasCdotc**：共轭的复数点积，result = Σ(conj(x[i]) * y[i])

广泛应用于信号处理、统计学、量子计算和线性代数等领域。

## 产品支持情况

| 产品                                                         |  是否支持 | 架构 |
| :----------------------------------------------------------- |:-------:|:----:|
| Ascend 950PR / Ascend 950DT                                  |    ✓    | arch35 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品                    |    ✓    | arch22 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品                    |    ✓    | arch22 |

## 目录结构介绍

```
blas/dot/
├── README.md                       // 说明文档
├── arch22/
│   ├── cdot_host.cpp               // Cdot Host 侧实现（arch22）
│   ├── cdot_kernel.cpp             // Cdot Kernel 侧实现（arch22）
│   ├── sdot_host.cpp               // Sdot Host 侧实现（arch22）
│   └── sdot_kernel.cpp             // Sdot Kernel 侧实现（arch22）
└── arch35/
    ├── sdot_host.cpp               // Sdot Host 侧实现（arch35）
    ├── sdot_kernel.cpp             // Sdot Kernel 侧实现（arch35）
    └── sdot_tiling_data.h          // Sdot Tiling 数据结构（arch35）

test/dot/sdot/
├── CMakeLists.txt                  // 测试构建文件
├── sdot_param.h                    // CSV 参数解析结构体
├── sdot_golden.h                   // CPU golden（cblas_sdot）
└── arch35/
    ├── sdot_test.cpp               // 测试代码（arch35，GTest 参数化）
    ├── sdot_test.csv               // CSV 测试用例
    └── sdot_npu_wrapper.h          // NPU wrapper
```

## 算子描述

### Sdot（实数点积）

- 算子功能：

sdot算子实现了两个实数向量的点积运算。对应的数学表达式为：
```
result = x · y = Σ(x[i] * y[i])  for i = 0 to n-1
```
其中x和y是实数向量。

- 对应的接口为：
```cpp
aclblasStatus_t aclblasSdot(aclblasHandle_t handle, const int64_t n, const float *x, const int64_t incx,
                              const float *y, const int64_t incy, float *result);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">sdot 参数说明</td>
   </tr>
<tr>
      <td rowspan="8" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">handle</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">ACL BLAS句柄，通过handle管理stream和workspace。</td>
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
      <td align="center">实数向量 float 指针（device侧），包含 n 个float元素。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">向量x的步长，不可为0。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">实数向量 float 指针（device侧），包含 n 个float元素。</td>
   </tr>
   <tr>
      <td align="center">incy</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">向量y的步长，不可为0。</td>
   </tr>
   <tr>
      <td align="center">result</td>
      <td align="center">device</td>
      <td align="center">out</td>
      <td align="center">实数结果 float 指针（device侧），包含 1 个float元素。</td>
   </tr>
</table>

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Sdot</td></tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">result</td><td align="center">1</td><td align="center">float</td><td align="center">ND</td></tr>
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

    **arch35 实现说明**：Host 侧通过 `GetAivCoreCount()` 获取 AIV 核数，workspace 从 handle 获取（`aclblasGetEffectiveWorkspace`），异步 launch kernel 后直接返回。Tiling 数据通过值传递给 kernel（结构体仅含 4 个字段，适合传值）。

### Cdot（复数点积）

- 算子功能：

**aclblasCdotu（无共轭点积）：**
```
result = Σ(x[i] * y[i])
其中：(a + bi) * (c + di) = (ac - bd) + (ad + bc)i
```

**aclblasCdotc（共轭点积）：**
```
result = Σ(conj(x[i]) * y[i])
其中：conj(x) = x_real - x_imag*i
      (a - bi) * (c + di) = (ac + bd) + (ad - bc)i
```

- 对应的接口为：

**aclblasCdotu接口：**
```cpp
int aclblasCdotu(const float *x, const float *y, float *result, const int64_t n, void *stream);
```

**aclblasCdotc接口：**
```cpp
int aclblasCdotc(const float *x, const float *y, float *result, const int64_t n, void *stream);
```

**参数说明：**
| 参数 | 内存位置 | 输入/输出 | 含义 |
|------|---------|----------|------|
| x | device | in | 复数向量，包含 2*n 个float元素（实部和虚部交替存储） |
| y | device | in | 复数向量，包含 2*n 个float元素（实部和虚部交替存储） |
| result | device | out | 复数结果，包含 2 个float元素（实部和虚部） |
| n | - | in | 复数向量的元素个数 |
| stream | - | in | ACL stream |

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Cdot</td></tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">2 * N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">2 * N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">result</td><td align="center">2</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cdot_kernel</td></tr>
  </table>

- 算子实现：

    1. **数据搬运**：将输入数据从x和y的GM地址分块搬运到UB
    2. **复数分离**：使用`vreducev2`指令将复数的实部和虚部分离
    3. **共轭处理**：cdotu (isConj=0) 不进行共轭处理；cdotc (isConj=1) 对x的虚部取反
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

在仓库根目录下执行如下步骤，编译并运行算子测试。

- 配置环境变量
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- 编译并执行测试
  ```bash
  bash build.sh --ops=sdot --run
  bash build.sh --ops=cdot --run
  ```

  执行结果如下，说明所有测试用例通过。
  ```bash
  [  PASSED  ] N tests.
  ```

## 算子特性

### 性能特性
- **向量化计算**：使用底层向量化指令（mul_v、cadd_v、vreducev2）实现高性能计算
- **流水线并行**：采用ping-pong双缓冲技术，实现数据搬运和计算的重叠
- **多核并行**：支持多核并行计算，自动分配计算任务

### 精度要求
- 相对误差 < 1e-5
