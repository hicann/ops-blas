# Nrm2算子

## 算子概述

向量范数算子，计算向量的欧几里得范数（2-范数），常用于向量长度计算、归一化和误差估计。nrm2 算子包含实数向量欧几里得范数（Snrm2）与复数向量欧几里得范数（Scnrm2）两个接口，是 BLAS 基础线性代数库中的核心算子之一。

数学表达式：

```
result = sqrt(sum(|x[i]|^2)) for i = 0 to n-1
```

复数向量（Scnrm2）：

```
result = sqrt(sum(|z[i]|^2)) = sqrt(sum(real[i]^2 + imag[i]^2))  for i = 0 to n-1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSnrm2 | 实数向量欧几里得范数（arch22 / arch35 均支持；arch35 支持 SIMD+SIMT 双路径，arch22 仅支持 incx==1） |
| aclblasScnrm2 | 复数向量欧几里得范数（针对 arch22 复用 snrm2 kernel 实现，仅支持 incx==1） |

## 产品支持情况

| 产品                                                         |  是否支持 | 架构 |
| :----------------------------------------------------------- |:-------:|:----:|
| Ascend 950PR / Ascend 950DT                                  |    ✓    | arch35 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品                    |    ✓    | arch22 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品                    |    ✓    | arch22 |

## 目录结构介绍

```
blas/nrm2/
├── README.md                          // 说明文档
├── arch22/
│   ├── snrm2_host.cpp                 // Snrm2 / Scnrm2 Host 侧实现（arch22）
│   └── snrm2_kernel.cpp               // Nrm2AIV 模板类 + 单核汇总（arch22，仅 incx==1）
└── arch35/
    ├── snrm2_host.cpp                 // Snrm2 Host 侧实现 + 双路径选择（arch35）
    ├── snrm2_kernel.cpp               // Snrm2AIV（SIMD）/ Snrm2SimtCompute（SIMT）/ Snrm2Reduce（汇总）
    └── snrm2_kernel.h                 // TilingData 结构体（Host 和 Kernel 共用）
```

## 算子执行接口

### aclblasSnrm2

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持（arch35）
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（arch22）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22）

#### 函数原型

```cpp
aclblasStatus_t aclblasSnrm2(aclblasHandle_t handle, int n, const float* x, int incx, float* result);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，内部携带 stream 和 workspace，Host 内存 |
| n | 输入 | int | 向量元素个数，n >= 0（n <= 0 时直接返回 0.0），Host 内存 |
| x | 输入 | const float*（FP32） | 输入向量，包含 n 个元素，当 n > 0 时不可为 nullptr，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长；incx == 1 时走 SIMD 路径，incx != 1 时走 SIMT 路径（incx <= 0 时直接返回 0.0），Host 内存 |
| result | 输出 | float*（FP32） | 输出标量，存储欧几里得范数计算结果，不可为 nullptr，Device 内存 |

#### 约束说明

- n >= 0
- incx != 0

#### 算子规格

<table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Snrm2</td></tr>
  <tr><td rowspan="1" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">n</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">result</td><td align="center">1</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">snrm2_aiv_kernel / snrm2_simt_kernel / snrm2_reduce_kernel</td></tr>
</table>

#### 算子实现

根据 `incx` 参数自动选择不同的计算路径，采用双路径 dispatch 策略（arch35；arch22 仅支持 incx == 1 的 SIMD 路径）：

- **incx == 1（SIMD 路径）**：多核均分向量元素，每核使用 DataCopy + Mul + ReduceSum 流水线计算局部平方和。UB 内单次搬运 chunk 个 float 元素，通过 256B 级归约得到 per-chunk 部分和。
- **incx != 1（SIMT 路径，仅 arch35）**：多核 SIMT 线程块（128-2048 线程）按 incx 跨步遍历向量，每线程累加局部平方和。通过 `asc_syncthreads` 同步后二叉树归约得到块级部分和。
- **汇总阶段**：各核将部分和写入 workspace，单核执行 ReduceSum + Sqrt 得到最终结果。

多核并行策略：元素维度均匀分配到多个 AIV Core，余数核多处理 1 个元素。使用内核调用符 <<<>>> 调用核函数。

#### 调用示例

示例代码仅供参考，具体编译和执行过程请参考[编译与运行样例](compile_and_run_example.md)。

### aclblasScnrm2

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（arch22）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22）

#### 函数原型

```cpp
aclblasStatus_t aclblasScnrm2(aclblasHandle_t handle, const int64_t n, uint8_t* x, const int64_t incx, uint8_t* result);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ACL 流 handle，用于传入 stream，Host 内存 |
| n | 输入 | int64_t | 复数元素个数（kernel 内部处理 2*n 个 float 元素），Host 内存 |
| x | 输入 | uint8_t*（FP32 complex） | 复数向量（交错实部/虚部存储，实际为 2*n 个 float），Device 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长（arch22 仅支持 incx == 1），Host 内存 |
| result | 输出 | uint8_t*（FP32） | 复数向量的欧几里得范数（float 结果，通过 uint8_t* 传出），Device 内存 |

#### 约束说明

- n >= 0
- incx != 0

#### 算子规格

<table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Scnrm2</td></tr>
  <tr><td rowspan="1" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">2*n</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">result</td><td align="center">1</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">snrm2_kernel</td></tr>
</table>

#### 算子实现

将复数向量的 2*n 个 float 元素直接传入 snrm2 kernel，复用实数向量范数计算逻辑。arch22 仅支持 incx == 1 的 SIMD 路径：按 32 元素块分配到多个 AIV Core，每核在 UB 内计算 local 平方和，通过 `SetAtomicAdd` 原子累积到 workspace，最后由 core 0 汇总计算 Sqrt。使用内核调用符 <<<>>> 调用核函数。

#### 调用示例

暂无示例代码，编译与运行流程请参考[编译与运行样例](compile_and_run_example.md)。

## 编译运行

在仓库根目录下执行如下步骤，编译并运行算子测试。

- 配置环境变量
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- Snrm2 编译并执行测试
  ```bash
  bash build.sh --ops=snrm2 --soc=ascend950 --run
  ```

  执行结果如下，说明所有测试用例通过。
  ```bash
  [Success] Case accuracy is verification passed.
  ```
