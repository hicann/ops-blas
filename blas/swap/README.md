## Swap算子实现

## 概述

BLAS Swap算子实现，包含实数向量交换（Sswap）和复数向量交换（Cswap）。

Swap算子实现了两个向量对应元素的交换操作，是BLAS基础线性代数库中的核心算子之一。

该算子包含以下接口：
- **aclblasSswap**：实数向量交换
- **aclblasCswap**：复数向量交换

swap 是 BLAS Level-1 函数，属于纯数据搬运类算子，不涉及任何数值计算。

## 产品支持情况

| 产品                                                         |  是否支持 | 架构 |
| :----------------------------------------------------------- |:-------:|:----:|
| Ascend 950PR / Ascend 950DT                                  |    ✓    | arch35 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品                    |    ✓    | arch22 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品                    |    ✓    | arch22 |

## 目录结构介绍

```
blas/swap/
├── README.md                       // 说明文档
├── arch22/
│   ├── cswap_host.cpp              // Cswap Host 侧实现（arch22）
│   ├── cswap_kernel.cpp            // Cswap Kernel 侧实现（arch22）
│   ├── sswap_host.cpp              // Sswap Host 侧实现（arch22）
│   └── sswap_kernel.cpp            // Sswap Kernel 侧实现（arch22）
└── arch35/
    ├── sswap_host.cpp              // Sswap Host 侧实现（arch35）
    ├── sswap_kernel.cpp            // Sswap Kernel 侧实现（arch35）
    └── sswap_tiling_data.h         // Sswap Tiling 数据结构（arch35）
```

## 算子描述

### Sswap（实数向量交换）

- 算子功能：

sswap 实现了两个实数向量对应元素的交换操作：
```
对于 i = 0, 1, ..., n-1:
  temp = x[i * incx]
  x[i * incx] = y[i * incy]
  y[i * incy] = temp
```

- 对应的接口为：

```cpp
aclblasStatus_t aclblasSswap(
    aclblasHandle_t handle,
    const int64_t n,
    uint8_t* x,
    const int64_t incx,
    uint8_t* y,
    const int64_t incy);
```

| 参数 | in/out | 设备 | 类型 | 含义 |
|------|--------|------|------|------|
| handle | in | host | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream |
| n | in | host | const int64_t | 向量中参与交换的元素个数 |
| x | in/out | device | uint8_t* | 指向 float 向量的 device 指针，交换后包含原 y 的元素 |
| incx | in | host | const int64_t | 向量 x 中相邻元素之间的步长 |
| y | in/out | device | uint8_t* | 指向 float 向量的 device 指针，交换后包含原 x 的元素 |
| incy | in | host | const int64_t | 向量 y 中相邻元素之间的步长 |

**注意**：x、y 必须为 device 侧指针，由调用者在调用前通过 `aclrtMalloc` 分配并通过 `aclrtMemcpy` 拷入数据。stream 通过 `aclblasSetStream(handle, stream)` 绑定到 handle。

- 支持规格：

| 项目 | 内容 |
|------|------|
| 数据类型 | FP32 (float) |
| 精度要求 | 位精确（Bitwise Match），MARE=0, MERE=0 |

- 参数约束：

| 条件 | 返回值 | 说明 |
|------|--------|------|
| `n <= 0` | `ACLBLAS_STATUS_SUCCESS` | 直接返回成功，不执行任何操作 |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | handle 空指针校验 |
| `x == nullptr \|\| y == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` | 数据指针空指针校验 |
| `incx == 0 \|\| incy == 0` | `ACLBLAS_STATUS_INVALID_VALUE` | 步长不能为零 |

- 算子实现：

  **多核切分**：将 `n` 个元素按 `ELEMENTS_PER_BLOCK` (8) 对齐后均匀分配给所有 Vector Core，最后一个核吸收剩余元素。

  **UB 切分与流水线**：
  - 采用 Single Buffer 模式（swap 无 Vector 计算阶段，Double Buffer 无额外收益）
  - 使用 `DataCopyPad` 统一处理完整 tile 和尾部 tile

  **交叉写回**：
  ```
  CopyIn:  xGM → inQueueX (UB)    CopyOut:  inQueueX (UB) → yGM  (x 数据写到 y 位置)
           yGM → inQueueY (UB)              inQueueY (UB) → xGM  (y 数据写到 x 位置)
  ```

  **arch22 与 arch35 实现差异**：

  | 项目 | arch22 实现 | arch35 实现 |
  |------|-----------|-------------|
  | TilingData | `startOffset[40]` / `calNum[40]` 数组 | `totalN` / `perCoreN` / `remainder` / `tileSize` 结构 |
  | 核数获取 | 硬编码 `numBlocks = 8` | `aclrtGetDeviceInfo` 动态获取 |
  | 编程风格 | 手写 `gm_to_ub_align` / `ub_to_gm_align` | 标准 AscendC `DataCopyPad` + `TPipe`/`TQue` |
  | 同步机制 | 手写 `SET_FLAG` / `WAIT_FLAG` 宏 | 标准 `SetFlag<HardEvent::MTE2_MTE3>` / `WaitFlag` API |

### Cswap（复数向量交换）

- 算子功能：

cswap算子实现了两个复数向量x和y的交换。对应的数学表达式为：
```
x <-> y
```

复数向量在内存中以交错float数组形式存储：[real0, imag0, real1, imag1, ...]

- 对应的接口：
```cpp
int aclblasCswap(aclblasHandle handle, float *x, float *y, const int64_t n, const int64_t incx, const int64_t incy);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">cswap 参数说明</td>
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
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">ACL流handle，用于传入stream。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">复数向量（存储为float数组，2*n个元素）。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">复数向量（存储为float数组，2*n个元素）。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">向量中的复数元素个数。</td>
   </tr>
   <tr>
      <td align="center">incx/incy</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">x/y中连续元素之间的步长。</td>
   </tr>
</table>

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Cswap</td></tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x/y</td><td align="center">2 * N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">算子输出</td><td align="center">x</td><td align="center">2 * N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">2 * N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cswap_kernel</td></tr>
  </table>

- 算子实现：

    复数向量被视为2*n个float元素，直接复用swap逻辑，使用ping-pong双缓冲策略完成交换。

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
  bash build.sh --ops=sswap --soc=ascend950 --run
  bash build.sh --ops=cswap --run
  ```

  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```
