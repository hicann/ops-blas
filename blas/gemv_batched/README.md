## GemvBatched 算子实现

## 概述

BLAS GemvBatched 算子实现。

GemvBatched（批量矩阵-向量乘法）实现了对一批矩阵分别进行矩阵-向量乘法的运算，是 BLAS Level 2 核心算子之一。

该算子包含以下接口：
- **aclblasSgemvBatched / aclblasHSHgemvBatched / aclblasHSSgemvBatched**：实数批量矩阵-向量乘法，针对 arch35（Ascend 950）架构，支持 S（FP32 入/出）、HSH（FP16 入/出）、HSS（FP16 入/FP32 出）三种精度
- **aclblasCgemvBatched**：复数批量矩阵-向量乘法，针对 arch22（Atlas A2/A3）架构

## 产品支持情况

| 产品                                                         |  是否支持 | 架构 |
| :----------------------------------------------------------- |:-------:|:----:|
| Ascend 950PR / Ascend 950DT                                  |    ✓    | arch35 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品                    |    ✓    | arch22 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品                    |    ✓    | arch22 |

## 目录结构介绍

```
blas/gemv_batched/
├── README.md                           // 说明文档
├── arch22/
│   ├── cgemv_batched_host.cpp          // CgemvBatched Host 侧实现（arch22）
│   ├── cgemv_batched_kernel.cpp        // CgemvBatched Kernel 侧实现（arch22）
│   ├── cgemv_batched_kernel_impl.h     // CgemvBatched Kernel 实现细节（arch22）
│   └── cgemv_batched_plan.h            // CgemvBatched 执行计划（arch22）
└── arch35/
    ├── gemv_batched_host.cpp           // Host 侧实现（参数校验、Tiling 计算、Kernel 调用）
    ├── gemv_batched_kernel.cpp         // Kernel 侧实现（AIV SIMD 批量运算 + SIMT 转置路径）
    └── gemv_batched_tiling_data.h      // Tiling 数据结构（Host 和 Kernel 共用）
```

测试代码位于 `test/gemv_batched/`：

```
test/gemv_batched/
├── CMakeLists.txt                      // 编译工程文件
├── gemv_batched_param.h                // CSV 参数解析
├── gemv_batched_golden.h               // CPU golden（调用 cblas_sgemv）
└── arch35/
    ├── gemv_batched_test.cpp           // GTest 精度测试
    ├── gemv_batched_test.csv           // CSV 测试用例
    └── gemv_batched_npu_wrapper.h      // NPU 调用封装
```

## 算子描述

### 实数批量矩阵-向量乘法（SgemvBatched / HSHgemvBatched / HSSgemvBatched）

- 算子功能：

GemvBatched 对每个 batch 独立完成矩阵-向量乘法。对应的数学表达式为：

```
y[i] = alpha * op(A[i]) * x[i] + beta * y[i]
```

其中 `op(A)` 可以是：
- `A`（不转置，trans = N）：维度 m×n，x 长度 n，y 长度 m
- `A^T`（转置，trans = T）：x 长度 m，y 长度 n

矩阵 A 采用列主序（column-major）存储。

- 对应的接口为：

```cpp
aclblasStatus_t aclblasSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans,
    int m, int n, const float *alpha, const float *A, int lda,
    const float *x, int incx, const float *beta,
    float *y, int incy, int batchCount);

aclblasStatus_t aclblasHSHgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans,
    int m, int n, const float *alpha, const uint16_t *A, int lda,
    const uint16_t *x, int incx, const float *beta,
    uint16_t *y, int incy, int batchCount);

aclblasStatus_t aclblasHSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans,
    int m, int n, const float *alpha, const uint16_t *A, int lda,
    const uint16_t *x, int incx, const float *beta,
    float *y, int incy, int batchCount);
```

| Param.        | Memory | in/out | 含义 |
| :------------ | :----- | :----: | :--- |
| handle        | Host   | in     | ops-blas 库上下文句柄 |
| trans         | Host   | in     | 矩阵操作类型：ACLBLAS_OP_N / ACLBLAS_OP_T |
| m, n          | Host   | in     | 矩阵 A 的行数 / 列数 |
| alpha         | Host   | in     | 标量乘数 |
| A             | Device | in     | 矩阵 A 数组（batch×m×n 行主序） |
| lda           | Host   | in     | A 矩阵的 leading dimension |
| x             | Device | in     | 向量 x 数组 |
| incx          | Host   | in     | x 向量元素步长 |
| beta          | Host   | in     | 标量乘数 |
| y             | Device | in/out | 向量 y 数组 |
| incy          | Host   | in     | y 向量元素步长 |
| batchCount    | Host   | in     | 批量大小 |

- 算子规格：

  <table>
  <tr><td rowspan="1" align="center">算子类型</td><td colspan="3" align="center">SgemvBatched / HSHgemvBatched / HSSgemvBatched</td></tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">A</td><td align="center">batch × m × n</td><td align="center">float / uint16_t</td></tr>
  <tr><td align="center">x</td><td align="center">batch × (trans=N: n, trans=T: m)</td><td align="center">float / uint16_t</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">batch × (trans=N: m, trans=T: n)</td><td align="center">float / uint16_t / float</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="3" align="center">gemv_batched</td></tr>
  </table>

- 算子实现：

  Host 侧通过 `GetAivCoreCount()` 获取 AIV 核数，Tiling 数据通过值传递给 kernel（无需 GM 分配）。Workspace 使用 handle 的默认 workspace（通过 `aclblasGetEffectiveWorkspace()` 获取），无需手动申请。Host 侧为异步执行，不包含流同步操作，由调用方负责同步。日志输出使用 `OP_LOG*` 宏（如 `OP_LOGD`、`OP_LOGE`）。

  trans=N（不转置）：
    - 使用 AIV SIMD 向量指令实现行级点积（VEC_SCOPE），支持 m-tiling 和 n-tiling 分片策略
    - 多核并行：按 batch 数均匀分配到多个 AIV Core

  trans=T（转置）：
    - incx=1 且 incy=1 时使用 AIV SIMD 双缓冲流水线
    - 其他步长组合使用 SIMT 编程模型，每个线程处理一个输出元素

- 调用实现：
  使用 `gemv_batched_kernel_do()` 封装内核调用。

### 复数批量矩阵-向量乘法（CgemvBatched）

- 算子功能：

CgemvBatched算子实现了批量复数矩阵与向量的乘法。对应的数学表达式为：
```
y[i] = A[i] * x[i]        (trans = N)
y[i] = A[i]^T * x[i]      (trans = T)
y[i] = A[i]^H * x[i]      (trans = C)
```
其中A[i]是复数矩阵，x[i]和y[i]是复数向量，i是批次索引。

复数乘法公式：(a + bi) * (c + di) = (ac - bd) + (ad + bc)i

- 对应的接口为：
```cpp
int aclblasCgemvBatched(const void *A, const void *x, void *y,
                        const int64_t batchCount, const int64_t m, const int64_t n,
                        const int32_t trans, const int32_t dtype,
                        void *stream);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">cgemv_batched 参数说明</td>
   </tr>
   <tr>
      <td rowspan="9" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">batchCount</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">批次数。</td>
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
      <td align="center">A</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">批量复数矩阵，batchCount个m×n矩阵。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">批量复数向量，batchCount个向量。</td>
   </tr>
   <tr>
      <td align="center">trans</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵操作类型：0=N(普通)，1=T(转置)，2=C(共轭转置)。</td>
   </tr>
   <tr>
      <td align="center">dtype</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">数据类型：0=half，1=float。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">device</td>
      <td align="center">out</td>
      <td align="center">批量复数向量，batchCount个向量。</td>
   </tr>
</table>

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">CgemvBatched</td></tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A, x</td><td align="center">batchCount × m × n</td><td align="center">complex&lt;float&gt;</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">batchCount × m</td><td align="center">complex&lt;float&gt;</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cgemv_batched_kernel</td></tr>
  </table>

- 算子实现：

    将输入数据从A和x的GM地址分块搬运到UB，使用GatherMask分离实部和虚部，进行复数矩阵-向量乘法计算，最后归约并搬出到y所在的GM地址。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 测试用例覆盖

测试基于 GTest + CSV 驱动框架，golden 实现调用 Netlib BLAS `cblas_sgemv`（逐 batch 调用）。

| 分组 | 用例数 | 覆盖场景 |
|------|--------|----------|
| L0 | 5 | FP32/FP16 Normal、FP16 Large/LargeM、FP32 Transpose |
| L1 | 54 | FP32/FP16/HSS/TST/TSS Normal+Transpose、大规模 m/n/batch、非对齐维度、多步长 lda/incx/incy、负步长 |
| Error | 11 | 无效 trans enum、m/n<0、lda<m、incx/incy=0、A/x/y/alpha/beta 空指针 |

## 编译运行

在仓库根目录下执行如下步骤，编译并运行算子测试。

- 配置环境变量
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- 编译算子
  ```bash
  bash build.sh --ops=gemv_batched --soc=ascend950
  bash build.sh --ops=cgemv_batched --run
  ```

- 编译并运行测试
  ```bash
  bash build.sh --ops=gemv_batched --soc=ascend950 --run
  ```
